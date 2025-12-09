from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main():
    spark = (
        SparkSession.builder
        .appName("NYC Taxi ZScore + MinMaxScale Demo")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------------
    # 1. Load NYC Yellow Taxi data (January 2023)
    # ------------------------------------------------------------------
    # Adjust the path if you put the file somewhere else
    taxi_path = "datasets/yellow_tripdata_2025-10.parquet"

    df_raw = spark.read.parquet(taxi_path)

    print("\n=== Raw schema (truncated) ===")
    df_raw.printSchema()

    # ------------------------------------------------------------------
    # 2. Light cleaning / filtering
    # ------------------------------------------------------------------
    # Keep only trips with:
    #   - positive distance
    #   - positive fare
    df = (
        df_raw
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("fare_amount") > 0)
        .filter(F.col("VendorID").isNotNull())
    )

    print("\n=== Row counts ===")
    print(f"Raw count:    {df_raw.count()}")
    print(f"Cleaned count:{df.count()}")

    # ------------------------------------------------------------------
    # 3. Per-VendorID z-score on trip_distance, AND global z-score
    # ------------------------------------------------------------------
    
    df_with_z = (
        df
        .withColumn(
            "trip_distance_z_vendor",
            F.zscore("trip_distance", "VendorID")  # per-VendorID
        )
        .withColumn(
            "trip_distance_z_global",
            F.zscore("trip_distance")              # global across all vendors
        )
    )

    # For display: top 3 longest trips per VendorID
    w_top = Window.partitionBy("VendorID").orderBy(F.col("trip_distance").desc())

    demo_sample = (
        df_with_z
        .withColumn("rank_within_vendor", F.row_number().over(w_top))
        .filter(F.col("rank_within_vendor") <= 3)
        .select(
            "VendorID",
            "trip_distance",
            "fare_amount",
            "trip_distance_z_vendor",
            "trip_distance_z_global",
        )
        .orderBy("VendorID", F.col("trip_distance").desc())
    )

    print("\n=== Sample (per-vendor vs global z-score) – top 3 trips per VendorID ===")
    demo_sample.show(30, truncate=False)
    
    # ------------------------------------------------------------------
    # 4. Filter "unusual" trips: |z| > 2.5
    # ------------------------------------------------------------------
    z_threshold = 2.5
    outliers = df_with_z.filter(F.abs(F.col("trip_distance_z_global")) > z_threshold)

    total_count = df_with_z.count()
    outlier_count = outliers.count()

    print("\n=== Outlier summary (|z| > 2.5 per VendorID) ===")
    print(f"Total cleaned trips:   {total_count}")
    print(f"Trips with |z| > 2.5:  {outlier_count}")
    if total_count > 0:
        print(f"Outlier ratio:         {outlier_count / total_count:.4%}")
        print("If this distribution was normalized, the outlier ratio for 2.5 would be about 1.24%")

    print("\n=== Example 'unusual distance' trips (largest |z| first) ===")
    outliers.orderBy(F.abs("trip_distance_z_global").desc()).select(
        "VendorID",
        "trip_distance",
        "fare_amount",
        "trip_distance_z_global"
    ).show(20, truncate=False)

    # ------------------------------------------------------------------
    # 5. Per-VendorID min-max scaling on BOTH distance and fare for outliers
    #    (to check how well scaled distance and scaled fare correlate)
    # ------------------------------------------------------------------
    
    # First: compute global min-max scaled distance and fare over ALL trips
    global_scaled = (
        df_with_z
        .withColumn(
            "distance_scaled_0_100",
            F.min_max_scale("trip_distance", 0.0, 100.0)  # global scaling
        )
        .withColumn(
            "fare_scaled_0_100",
            F.min_max_scale("fare_amount", 0.0, 100.0)    # global scaling
        )
    )

    # Reuse the same "top 3 longest trips per VendorID" logic from Step 3
    w_top = Window.partitionBy("VendorID").orderBy(F.col("trip_distance").desc())

    demo_scaled = (
        global_scaled
        .withColumn("rank_within_vendor", F.row_number().over(w_top))
        .filter(F.col("rank_within_vendor") <= 3)
        .select(
            "VendorID",
            "trip_distance",
            "fare_amount",
            "distance_scaled_0_100",
            "fare_scaled_0_100",
        )
        .orderBy("VendorID", F.col("trip_distance").desc())
    )

    print("\n=== Global scaled distance vs global scaled fare (same rows as Step 3) ===")
    demo_scaled.show(30, truncate=False)
    
    spark.stop()


if __name__ == "__main__":
    main()

