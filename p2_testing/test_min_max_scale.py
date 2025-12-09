from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark
spark = SparkSession.builder \
    .appName("MinMaxScaleTest") \
    .master("local[*]") \
    .getOrCreate()

print("=" * 60)
print("PySpark min_max_scale Tests")
print("=" * 60)

# Test 1: Global scaling (no grouping)
print("\n1. Global scaling [0, 100]:")
df1 = spark.createDataFrame(
    [(10.0,), (20.0,), (30.0,), (40.0,)],
    ["value"]
)
df1.select(
    "value",
    F.min_max_scale("value", 0, 100).alias("scaled")
).show()

# Test 2: Per-group scaling
print("\n2. Per-group scaling:")
df2 = spark.createDataFrame(
    [("A", 10.0), ("A", 20.0), ("A", 30.0),
     ("B", 5.0), ("B", 15.0), ("B", 25.0)],
    ["group", "value"]
)
df2.select(
    "group",
    "value",
    F.min_max_scale("value", 0, 100, F.col("group")).alias("scaled")
).orderBy("group", "value").show()

# Test 3: Constant values (all identical)
print("\n3. Constant values (should map to midpoint 50):")
df3 = spark.createDataFrame(
    [("A", 10.0), ("A", 10.0), ("A", 10.0),
     ("B", 20.0), ("B", 20.0)],
    ["group", "value"]
)
df3.select(
    "group",
    "value",
    F.min_max_scale("value", 0, 100, F.col("group")).alias("scaled")
).orderBy("group").show()

# Test 4: Null handling
print("\n4. Null handling:")
df4 = spark.createDataFrame(
    [("A", 10.0), ("A", None), ("A", 30.0),
     ("B", 5.0), ("B", None), ("B", 15.0)],
    ["group", "value"]
)
df4.select(
    "group",
    "value",
    F.min_max_scale("value", 0, 100, F.col("group")).alias("scaled")
).orderBy("group", F.col("value").asc_nulls_first()).show()

# Test 5: Multiple grouping columns
print("\n5. Multiple grouping columns:")
df5 = spark.createDataFrame(
    [("A", "X", 10.0), ("A", "X", 20.0),
     ("A", "Y", 5.0), ("A", "Y", 15.0),
     ("B", "X", 100.0), ("B", "X", 200.0)],
    ["group1", "group2", "value"]
)
df5.select(
    "group1", "group2", "value",
    F.min_max_scale("value", 0, 1, F.col("group1"), F.col("group2")).alias("scaled")
).orderBy("group1", "group2", "value").show()

# Test 6: Default range [0, 1]
print("\n6. Default range [0, 1]:")
df6 = spark.createDataFrame(
    [(10.0,), (20.0,), (30.0,)],
    ["value"]
)
df6.select(
    "value",
    F.min_max_scale("value").alias("scaled")
).show()

# Test 7: Custom range [-1, 1]
print("\n7. Custom range [-1, 1]:")
df7 = spark.createDataFrame(
    [(10.0,), (20.0,), (30.0,)],
    ["value"]
)
df7.select(
    "value",
    F.min_max_scale("value", -1, 1).alias("scaled")
).show()

spark.stop()
