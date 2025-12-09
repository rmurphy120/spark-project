package com.p2honors.spark.sql

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import com.yourcompany.spark.sql.functions.min_max_scale

class MinMaxScaleSpec extends AnyFunSuite with BeforeAndAfterAll {
  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("MinMaxScaleSpec")
      .master("local[*]")
      .getOrCreate()
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  test("global scaling to [0, 100]") {
    import spark.implicits._
    val df = Seq(10.0, 20.0, 30.0).toDF("value")
    val result = df.select(min_max_scale(col("value"), 0.0, 100.0).as("scaled"))
      .collect()
      .map(_.getDouble(0))
    
    assert(result(0) === 0.0)
    assert(result(1) === 50.0)
    assert(result(2) === 100.0)
  }

  test("per-group scaling") {
    import spark.implicits._
    val df = Seq(
      ("A", 10.0), ("A", 20.0),
      ("B", 5.0), ("B", 15.0)
    ).toDF("group", "value")
    
    val result = df.select(
      col("group"),
      min_max_scale(col("value"), 0.0, 100.0, col("group")).as("scaled")
    ).collect()
    
    // Group A: 10->0, 20->100
    // Group B: 5->0, 15->100
    assert(result.length === 4)
  }

  test("constant values map to midpoint") {
    import spark.implicits._
    val df = Seq(("A", 10.0), ("A", 10.0)).toDF("group", "value")
    val result = df.select(min_max_scale(col("value"), 0.0, 100.0, col("group")).as("scaled"))
      .collect()
      .map(_.getDouble(0))
    
    assert(result.forall(_ === 50.0))
  }

  test("null values are preserved") {
    import spark.implicits._
    val df = Seq(Some(10.0), None, Some(20.0)).toDF("value")
    val result = df.select(min_max_scale(col("value"), 0.0, 100.0).as("scaled"))
      .collect()
    
    assert(result(1).isNullAt(0))
  }
}
