from ast import expr
import os
import sys
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, round as sql_round, when, udf, variance, expr
)
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, lag, lead, row_number

# Налаштування Python для PySpark
os.environ["PYSPARK_PYTHON"] = sys.executable

print("=== РІВЕНЬ 3: ПРОДВИНУТИЙ ===")
print("Ініціалізація SparkSession...")

start_time = time.time()

spark = SparkSession.builder \
    .appName("AirlineAnalysis_Level3") \
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

# --- Завдання 3.1: UDF та порівняння продуктивності ---
print("\nЗавдання 3.1: Реалізація UDF для категоризації затримок та порівняння з вбудованими функціями...")

# 1. Визначення UDF функції
def categorize_delay_func(minutes):
    if minutes is None:
        return "Unknown"
    elif minutes <= 0:
        return "No Delay"
    elif minutes <= 15:
        return "Short Delay"
    elif minutes <= 60:
        return "Medium Delay"
    elif minutes <= 180:
        return "Long Delay"
    else:
        return "Extreme Delay"

categorize_udf = udf(categorize_delay_func, StringType())

# Вимірюємо час виконання UDF
t_udf_start = time.time()
udf_df = df.withColumn("delay_category_udf", categorize_udf(col("DEP_DELAY")))
udf_count = udf_df.groupBy("delay_category_udf").count().collect()
t_udf_duration = time.time() - t_udf_start

print(f"Час виконання з Python UDF: {t_udf_duration:.3f} секунд")
for row in udf_count:
    print(f"  - {row['delay_category_udf']}: {row['count']:,}")

# 2. Вимірюємо час виконання з вбудованою функцією when().otherwise() (втричі швидше за UDF)
t_builtin_start = time.time()
builtin_df = df.withColumn(
    "delay_category_builtin",
    when(col("DEP_DELAY") <= 0, "No Delay")
    .when(col("DEP_DELAY") <= 15, "Short Delay")
    .when(col("DEP_DELAY") <= 60, "Medium Delay")
    .when(col("DEP_DELAY") <= 180, "Long Delay")
    .otherwise("Extreme Delay")
)
builtin_count = builtin_df.groupBy("delay_category_builtin").count().collect()
t_builtin_duration = time.time() - t_builtin_start

print(f"Час виконання з вбудованими функціями (when/otherwise): {t_builtin_duration:.3f} секунд")
if t_builtin_duration > 0:
    print(f"Прискорення вбудованих функцій порівняно з UDF: {t_udf_duration / t_builtin_duration:.2f}x")


# --- Завдання 3.2: Window Functions та тренд аналіз ---
print("\nЗавдання 3.2: Віконні функції (Window Functions) та тренд-аналіз...")

# Створюємо віконні специфікації
window_carrier_date = Window.partitionBy("OP_CARRIER").orderBy("FL_DATE")
window_7_day_rolling = Window.partitionBy("OP_CARRIER").orderBy("FL_DATE").rowsBetween(-6, 0)
window_carrier_month = Window.partitionBy("OP_CARRIER", expr("month(FL_DATE)")).orderBy(avg("DEP_DELAY"))

# 1. Обчислення ковзного середнього затримки за 7 днів та Lag/Lead
print("Застосування віконних функцій (Rolling Average та Lag/Lead)...")
window_features_df = df.withColumn(
    "rolling_avg_dep_delay", avg("DEP_DELAY").over(window_7_day_rolling)
).withColumn(
    "prev_day_delay", lag("DEP_DELAY", 1).over(window_carrier_date)
).withColumn(
    "next_day_delay", lead("DEP_DELAY", 1).over(window_carrier_date)
)

window_features_df.select("FL_DATE", "OP_CARRIER", "DEP_DELAY", "rolling_avg_dep_delay", "prev_day_delay").show(5, truncate=False)

# 2. Знаходження авіакомпанії з найбільш нестабільними затримками (highest variance)
print("\nАналіз варіативності затримок по авіакомпаніях:")
carrier_variance_df = df.groupBy("OP_CARRIER").agg(
    sql_round(variance("DEP_DELAY"), 2).alias("delay_variance"),
    sql_round(avg("DEP_DELAY"), 2).alias("avg_delay")
).orderBy(col("delay_variance").desc())

carrier_variance_df.show(truncate=False)

elapsed_time = time.time() - start_time
print(f"\n Загальний час виконання Рівня 3: {elapsed_time:.2f} секунд")

spark.stop()