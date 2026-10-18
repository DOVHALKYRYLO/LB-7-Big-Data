import os
import sys
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, max as sql_max, round as sql_round, 
    month, dayofweek, hour, when, expr
)

# Налаштування Python для PySpark
os.environ["PYSPARK_PYTHON"] = sys.executable

print("=== РІВЕНЬ 2: СТАНДАРТНИЙ ===")
print("Ініціалізація SparkSession...")

start_time = time.time()

spark = SparkSession.builder \
    .appName("AirlineAnalysis_Level2") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

csv_path = "data/2018.csv"

if os.path.exists(csv_path):
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
else:
    print(f"Файл {csv_path} не знайдено. Генеруємо симуляцію датасету...")
    np.random.seed(42)
    n_rows = 1500000
    sim_data = pd.DataFrame({
        'FL_DATE': pd.to_datetime(np.random.choice(pd.date_range('2018-01-01', '2018-12-31'), n_rows)),
        'OP_CARRIER': np.random.choice(['AA', 'DL', 'UA', 'WN', 'B6'], n_rows),
        'ORIGIN': np.random.choice(['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'SFO', 'MIA'], n_rows),
        'DEST': np.random.choice(['LAX', 'JFK', 'SFO', 'MIA', 'ORD', 'ATL', 'DFW'], n_rows),
        'DEP_DELAY': np.random.normal(10, 25, n_rows),
        'ARR_DELAY': np.random.normal(8, 28, n_rows),
        'DISTANCE': np.random.randint(200, 2500, n_rows),
        'DEP_TIME': np.random.randint(600, 2359, n_rows),
        'CANCELLED': np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    })
    os.makedirs("data", exist_ok=True)
    sim_data.to_csv(csv_path, index=False)
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)

# --- Завдання 2.1: Складні агрегації ---
print("\nЗавдання 2.1: Розрахунок складних агрегацій по аеропортах...")

airport_stats = df.groupBy("ORIGIN").agg(
    count("*").alias("total_flights"),
    sql_round(avg("DEP_DELAY"), 2).alias("avg_dep_delay"),
    sql_round(sql_max("ARR_DELAY"), 2).alias("max_arr_delay"),
    sql_round((count(when(col("CANCELLED") == 1, True)) * 100.0 / count("*")), 2).alias("cancellation_rate")
).withColumnRenamed("ORIGIN", "airport_code") \
 .orderBy(col("total_flights").desc())

print("Топ наиболее завантажених аеропортів (з показниками затримок та скасувань):")
airport_stats.show(20, truncate=False)

# --- Завдання 2.2: Часовий аналіз та Heat Map ---
print("\nЗавдання 2.2: Часовий аналіз та створення Heat Map матриці...")

# Додаємо часові колонки
time_df = df.withColumn("month", month(col("FL_DATE"))) \
            .withColumn("day_of_week", dayofweek(col("FL_DATE"))) # Понеділок=2/Неділя=1 у стандарті Spark, зсунемо для 1=Mon...7=Sun

# Перетворення dayofweek у стандарт: 1=Mon, ..., 7=Sun
time_df = time_df.withColumn("day_of_week_std", 
    when(col("day_of_week") == 1, 7).otherwise(col("day_of_week") - 1)
)

print("Середня кількість рейсів за місяцями:")
monthly_avg = time_df.groupBy("month").agg(count("*").alias("total_flights")) \
                     .orderBy("month")
monthly_avg.show(12)

print("Найбільш завантажений день тижня (1=Пн, 7=Нд):")
busy_dow = time_df.groupBy("day_of_week_std").agg(count("*").alias("total_flights")) \
                  .orderBy(col("total_flights").desc())
busy_dow.show()

# Створення Heat Map: місяць vs день_тижня
print("\n=== Heat Map матриця (Місяць vs День тижня) ===")
heatmap_df = time_df.groupBy("month").pivot("day_of_week_std", [1, 2, 3, 4, 5, 6, 7]).count()

# Перейменуємо колонки для гарного відображення
heatmap_formatted = heatmap_df.select(
    col("month").alias("Month"),
    col("1").alias("Mon"),
    col("2").alias("Tue"),
    col("3").alias("Wed"),
    col("4").alias("Thu"),
    col("5").alias("Fri"),
    col("6").alias("Sat"),
    col("7").alias("Sun")
).orderBy("Month")

heatmap_formatted.show(12, truncate=False)

elapsed_time = time.time() - start_time
print(f"\n Загальний час виконання Рівня 2: {elapsed_time:.2f} секунд")

spark.stop()