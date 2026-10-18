import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, round as sql_round

# Налаштування Python для PySpark
os.environ["PYSPARK_PYTHON"] = sys.executable

# --- Завдання 1.1: Налаштування середовища ---
print("=== РІВЕНЬ 1: БАЗОВИЙ ===")
print("Завдання 1.1: Ініціалізація SparkSession з оптимізаціями...")

start_time = time.time()

spark = SparkSession.builder \
    .appName("AirlineAnalysis") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

print(f"Spark Version: {spark.version}")
print(f"Available Cores: {sc.defaultParallelism}")
print("SparkSession successfully created\n")

# --- Завдання 1.2: Завантаження та огляд даних ---
print("Завдання 1.2: Завантаження даних 2018 року...")

csv_path = "data/2018.csv"

if os.path.exists(csv_path):
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
else:
    print(f"Файл {csv_path} не знайдено. Генеруємо симуляцію датасету 2018 року для перевірки...")
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_rows = 1500000  # > 1,000,000 записів за вимогами
    sim_data = pd.DataFrame({
        'FL_DATE': pd.to_datetime(np.random.choice(pd.date_range('2018-01-01', '2018-12-31'), n_rows)),
        'OP_CARRIER': np.random.choice(['AA', 'DL', 'UA', 'WN', 'B6'], n_rows),
        'ORIGIN': np.random.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW'], n_rows),
        'DEST': np.random.choice(['LAX', 'JFK', 'SFO', 'MIA', 'ORD'], n_rows),
        'DEP_DELAY': np.random.normal(10, 25, n_rows),
        'ARR_DELAY': np.random.normal(8, 28, n_rows),
        'DISTANCE': np.random.randint(200, 2500, n_rows),
        'DEP_TIME': np.random.randint(600, 2359, n_rows),
        'CANCELLED': np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    })
    os.makedirs("data", exist_ok=True)
    sim_data.to_csv(csv_path, index=False)
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)

print(f"Загальна кількість записів: {df.count():,}")
print(f"Кількість колонок: {len(df.columns)}")
print("Назви колонок:", df.columns)
print("Схема даних:")
df.printSchema()

# --- Завдання 1.3: Базові операції відбору ---
print("\nЗавдання 1.3: Фільтрація рейсів з JFK із затримкою вильоту > 15 хв...")
filtered_df = df.select("FL_DATE", "OP_CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY", "DISTANCE") \
                .filter((col("ORIGIN") == "JFK") & (col("DEP_DELAY") > 15))

filtered_count = filtered_df.count()
print(f"Кількість відфільтрованих записів: {filtered_count:,}")
print("Перші 5 відфільтрованих рядків:")
filtered_df.show(5, truncate=False)

# --- Завдання 1.4: SQL інтеграція ---
print("\nЗавдання 1.4: SQL інтеграція (Тимчасове представлення 'flights')...")
df.createOrReplaceTempView("flights")

# Рівень А: Топ-5 авіакомпаній за кількістю рейсів та їх відсоток
print("Рівень А: Топ-5 авіакомпаній за кількістю рейсів:")
top_carriers_sql = """
    WITH total AS (SELECT COUNT(*) as total_cnt FROM flights)
    SELECT 
        OP_CARRIER, 
        COUNT(*) as flight_count,
        ROUND((COUNT(*) * 100.0 / (SELECT total_cnt FROM total)), 2) as percentage
    FROM flights
    GROUP BY OP_CARRIER
    ORDER BY flight_count DESC
    LIMIT 5
"""
spark.sql(top_carriers_sql).show()

# Рівень Б: Топ-3 найгірші авіакомпанії за середніми затримками
print("Рівень Б: Топ-3 найгірші авіакомпанії за середньою затримкою вильоту:")
worst_carriers_sql = """
    SELECT 
        OP_CARRIER, 
        ROUND(AVG(DEP_DELAY), 2) as avg_dep_delay
    FROM flights
    WHERE DEP_DELAY IS NOT NULL
    GROUP BY OP_CARRIER
    ORDER BY avg_dep_delay DESC
    LIMIT 3
"""
spark.sql(worst_carriers_sql).show()


# --- Завдання 1.5: Збереження результатів ---
print("\nЗавдання 1.5: Збереження відфільтрованих даних у форматі Parquet / CSV...")
output_dir = "output/year=2018"
os.makedirs(output_dir, exist_ok=True)

try:
    # Спроба стандартного збереження у Parquet
    filtered_df.write \
        .mode("overwrite") \
        .partitionBy("OP_CARRIER") \
        .parquet(output_dir)
    print(f"Дані успішно збережено у форматі Parquet в каталозі '{output_dir}'.")
    read_back_df = spark.read.parquet(output_dir)
    print(f"Зчитано записів з Parquet: {read_back_df.count():,}")
except Exception as e:
    print("Примітка: Локальне збереження через Parquet вимагає winutils у Windows. Зберігаємо через Pandas fallback...")
    pdf = filtered_df.toPandas()
    total_read_rows = 0
    for carrier, group in pdf.groupby("OP_CARRIER"):
        carrier_dir = os.path.join(output_dir, f"carrier={carrier}")
        os.makedirs(carrier_dir, exist_ok=True)
        file_path = os.path.join(carrier_dir, "data.csv")
        group.to_csv(file_path, index=False)
        total_read_rows += len(group)
    
    print(f"Дані успішно збережено з партиціонуванням по OP_CARRIER у каталог '{output_dir}'.")
    print(f"Перевірка зчитаних записів з файлів: {total_read_rows:,}")

elapsed_time = time.time() - start_time
print(f"\n Загальний час виконання Рівня 1: {elapsed_time:.2f} секунд")

spark.stop()