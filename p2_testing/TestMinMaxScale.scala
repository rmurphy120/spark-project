package com.p2honors.spark.test

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import com.yourcompany.spark.sql.functions.min_max_scale

object TestMinMaxScale {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("MinMaxScaleTest")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    println("=" * 60)
    println("Scala min_max_scale Tests")
    println("=" * 60)

    // Test 1: Global scaling
    println("\n1. Global scaling [0, 100]:")
    val df1 = Seq(10.0, 20.0, 30.0, 40.0).toDF("value")
    df1.select(
      col("value"),
      min_max_scale(col("value"), 0.0, 100.0).as("scaled")
    ).show()

    // Test 2: Per-group scaling
    println("\n2. Per-group scaling:")
    val df2 = Seq(
      ("A", 10.0), ("A", 20.0), ("A", 30.0),
      ("B", 5.0), ("B", 15.0), ("B", 25.0)
    ).toDF("group", "value")
    df2.select(
      col("group"),
      col("value"),
      min_max_scale(col("value"), 0.0, 100.0, col("group")).as("scaled")
    ).orderBy("group", "value").show()

    // Test 3: Constant values
    println("\n3. Constant values (should map to midpoint 50):")
    val df3 = Seq(
      ("A", 10.0), ("A", 10.0), ("A", 10.0),
      ("B", 20.0), ("B", 20.0)
    ).toDF("group", "value")
    df3.select(
      col("group"),
      col("value"),
      min_max_scale(col("value"), 0.0, 100.0, col("group")).as("scaled")
    ).orderBy("group").show()

    // Test 4: Null handling
    println("\n4. Null handling:")
    val df4 = Seq(
      ("A", Some(10.0)), ("A", None), ("A", Some(30.0)),
      ("B", Some(5.0)), ("B", None), ("B", Some(15.0))
    ).toDF("group", "value")
    df4.select(
      col("group"),
      col("value"),
      min_max_scale(col("value"), 0.0, 100.0, col("group")).as("scaled")
    ).orderBy("group", col("value").asc_nulls_first).show()

    // Test 5: Multiple grouping columns
    println("\n5. Multiple grouping columns:")
    val df5 = Seq(
      ("A", "X", 10.0), ("A", "X", 20.0),
      ("A", "Y", 5.0), ("A", "Y", 15.0),
      ("B", "X", 100.0), ("B", "X", 200.0)
    ).toDF("group1", "group2", "value")
    df5.select(
      col("group1"), col("group2"), col("value"),
      min_max_scale(col("value"), 0.0, 1.0, col("group1"), col("group2")).as("scaled")
    ).orderBy("group1", "group2", "value").show()

    // Test 6: Default range [0, 1]
    println("\n6. Default range [0, 1]:")
    val df6 = Seq(10.0, 20.0, 30.0).toDF("value")
    df6.select(
      col("value"),
      min_max_scale(col("value")).as("scaled")
    ).show()

    // Test 7: String column overload
    println("\n7. String column name with grouping:")
    val df7 = Seq(
      ("A", 10.0), ("A", 20.0),
      ("B", 5.0), ("B", 15.0)
    ).toDF("group", "value")
    df7.select(
      col("group"),
      col("value"),
      min_max_scale("value", 0.0, 100.0, col("group")).as("scaled")
    ).orderBy("group", "value").show()

    spark.stop()
  }
}
