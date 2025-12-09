import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.Column

object TestMinMaxScaleNYCTaxi {
  
  // Your min_max_scale implementation
  def min_max_scale(
      e: Column,
      outputMin: Double,
      outputMax: Double,
      groupBy: Column*): Column = {
    val w = if (groupBy.nonEmpty) {
      Window.partitionBy(groupBy: _*)
    } else {
      Window.partitionBy()
    }

    val minCol = min(e).over(w)
    val maxCol = max(e).over(w)
    val range = maxCol - minCol

    val outputRange = outputMax - outputMin
    val midpoint = (outputMax + outputMin) / 2.0

    when(e.isNull, lit(null))
      .when(range === 0.0, lit(midpoint))
      .otherwise(
        ((e - minCol) / range) * lit(outputRange) + lit(outputMin)
      )
  }

  def min_max_scale(e: Column, groupBy: Column*): Column = {
    min_max_scale(e, 0.0, 1.0, groupBy: _*)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("NYC Taxi MinMaxScale Demo")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    // ------------------------------------------------------------------
    // 1. Load NYC Yellow Taxi data
    // ------------------------------------------------------------------
    val taxiPath = "datasets/yellow_tripdata_2025-10.parquet"

    val dfRaw = spark.read.parquet(taxiPath)

    println("\n=== Raw schema (truncated) ===")
    dfRaw.printSchema()

    // ------------------------------------------------------------------
    // 2. Light cleaning / filtering
    // ------------------------------------------------------------------
    val df = dfRaw
      .filter(col("trip_distance") > 0)
      .filter(col("fare_amount") > 0)
      .filter(col("VendorID").isNotNull)

    println("\n=== Row counts ===")
    println(s"Raw count:     ${dfRaw.count()}")
    println(s"Cleaned count: ${df.count()}")

    // ------------------------------------------------------------------
    // 3. Global min-max scaling on distance and fare
    // ------------------------------------------------------------------
    
    val globalScaled = df
      .withColumn(
        "distance_scaled_0_100",
        min_max_scale(col("trip_distance"), 0.0, 100.0)  // global scaling
      )
      .withColumn(
        "fare_scaled_0_100",
        min_max_scale(col("fare_amount"), 0.0, 100.0)    // global scaling
      )

    // Top 3 longest trips per VendorID
    val wTop = Window.partitionBy("VendorID").orderBy(col("trip_distance").desc)

    val demoGlobalScaled = globalScaled
      .withColumn("rank_within_vendor", row_number().over(wTop))
      .filter(col("rank_within_vendor") <= 3)
      .select(
        col("VendorID"),
        col("trip_distance"),
        col("fare_amount"),
        col("distance_scaled_0_100"),
        col("fare_scaled_0_100")
      )
      .orderBy(col("VendorID"), col("trip_distance").desc)

    println("\n=== Global scaled distance vs global scaled fare (top 3 trips per VendorID) ===")
    demoGlobalScaled.show(30, truncate = false)

    // ------------------------------------------------------------------
    // 4. Per-VendorID min-max scaling
    // ------------------------------------------------------------------
    
    val perVendorScaled = df
      .withColumn(
        "distance_scaled_per_vendor",
        min_max_scale(col("trip_distance"), 0.0, 100.0, col("VendorID"))
      )
      .withColumn(
        "fare_scaled_per_vendor",
        min_max_scale(col("fare_amount"), 0.0, 100.0, col("VendorID"))
      )

    val demoPerVendorScaled = perVendorScaled
      .withColumn("rank_within_vendor", row_number().over(wTop))
      .filter(col("rank_within_vendor") <= 3)
      .select(
        col("VendorID"),
        col("trip_distance"),
        col("fare_amount"),
        col("distance_scaled_per_vendor"),
        col("fare_scaled_per_vendor")
      )
      .orderBy(col("VendorID"), col("trip_distance").desc)

    println("\n=== Per-vendor scaled distance vs fare (top 3 trips per VendorID) ===")
    demoPerVendorScaled.show(30, truncate = false)

    // ------------------------------------------------------------------
    // 5. Compare global vs per-vendor scaling side by side
    // ------------------------------------------------------------------
    
    val comparison = df
      .withColumn("distance_global_scaled", min_max_scale(col("trip_distance"), 0.0, 100.0))
      .withColumn("distance_vendor_scaled", min_max_scale(col("trip_distance"), 0.0, 100.0, col("VendorID")))
      .withColumn("rank_within_vendor", row_number().over(wTop))
      .filter(col("rank_within_vendor") <= 3)
      .select(
        col("VendorID"),
        col("trip_distance"),
        col("distance_global_scaled"),
        col("distance_vendor_scaled")
      )
      .orderBy(col("VendorID"), col("trip_distance").desc)

    println("\n=== Global vs Per-Vendor scaling comparison (top 3 per VendorID) ===")
    comparison.show(30, truncate = false)

    // ------------------------------------------------------------------
    // 6. Summary statistics
    // ------------------------------------------------------------------
    
    println("\n=== Trip distance statistics by VendorID ===")
    df.groupBy("VendorID")
      .agg(
        count("*").as("trip_count"),
        min("trip_distance").as("min_distance"),
        max("trip_distance").as("max_distance"),
        avg("trip_distance").as("avg_distance")
      )
      .orderBy("VendorID")
      .show(truncate = false)

    println("\n=== Fare amount statistics by VendorID ===")
    df.groupBy("VendorID")
      .agg(
        count("*").as("trip_count"),
        min("fare_amount").as("min_fare"),
        max("fare_amount").as("max_fare"),
        avg("fare_amount").as("avg_fare")
      )
      .orderBy("VendorID")
      .show(truncate = false)

    // ------------------------------------------------------------------
    // 7. Test edge cases
    // ------------------------------------------------------------------
    
    println("\n=== Testing edge case: trips with identical distances per vendor ===")
    val edgeCaseDF = Seq(
      (1, 5.0, 20.0),
      (1, 5.0, 25.0),
      (1, 5.0, 30.0),
      (2, 10.0, 40.0),
      (2, 10.0, 45.0)
    ).toDF("VendorID", "trip_distance", "fare_amount")

    edgeCaseDF
      .withColumn("distance_scaled", min_max_scale(col("trip_distance"), 0.0, 100.0, col("VendorID")))
      .show(truncate = false)

    println("Note: When all values in a group are identical, they map to the midpoint (50.0)")

    spark.stop()
  }
}
