from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, percentile_approx, coalesce

spark = SparkSession.builder.appName("OpenAQBatch").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 1) Cargar
df_raw = (spark.read
          .option("header", True)
          .csv("data/air_quality_openaq.csv"))

# 2) Filtrar PM2.5 y tipificar
#   - "datetime" es la columna de tiempo
#   - por si el formato varía, probamos varios patrones con coalesce
ts = coalesce(
    to_timestamp(col("datetime")),
    to_timestamp(col("datetime"), "yyyy-MM-dd'T'HH:mm:ss'Z'"),
    to_timestamp(col("datetime"), "yyyy-MM-dd HH:mm:ss"),
    to_timestamp(col("datetime"), "yyyy/MM/dd HH:mm:ss")
)

df = (df_raw
      .filter(col("parameter").isin("pm25", "PM2.5", "pm2.5"))
      .withColumn("station_id", col("location"))
      .withColumn("timestamp", ts)
      .withColumn("pm25", col("value").cast("double"))
      .filter(col("timestamp").isNotNull() & col("station_id").isNotNull())
      .filter((col("pm25") >= 0) & (col("pm25") <= 1000))
      .withColumn("hour", hour(col("timestamp")))
      .withColumn("dow", dayofweek(col("timestamp")))
)

# 3) Salidas
df.write.mode("overwrite").parquet("output/cleansed")

hourly = (df.groupBy("station_id", "hour").agg({"pm25":"avg"}))
hourly.write.mode("overwrite").parquet("output/aggregates_hourly")

p90 = (df.groupBy("station_id")
         .agg(percentile_approx("pm25", 0.90).alias("pm25_p90")))
p90.write.mode("overwrite").parquet("output/aggregates_percentiles")

spark.stop()
