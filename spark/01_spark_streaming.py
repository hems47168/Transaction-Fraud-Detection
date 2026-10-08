from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType
)
from pyspark.sql.functions import (
    from_json,
    col,
    when,
    to_json,
    struct
)
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array


# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("RealTimeFraudDetectionStreaming") \
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Spark Streaming application started!")


# ============================================================
# 2. CONNECT TO KAFKA
# ============================================================

kafka_df = spark.readStream \
    .format("kafka") \
    .option(
        "kafka.bootstrap.servers",
        "localhost:9092"
    ) \
    .option(
        "subscribe",
        "bank_transactions"
    ) \
    .option(
        "startingOffsets",
        "latest"
    ) \
    .load()

print("Connected to Kafka successfully!")


# ============================================================
# 3. LOAD TRAINED RANDOM FOREST MODEL
# ============================================================

model = PipelineModel.load(
    "model/fraud_model"
)

print("Fraud detection model loaded successfully!")


# ============================================================
# 4. DEFINE TRANSACTION SCHEMA
# ============================================================

transaction_schema = StructType([
    StructField(
        "transaction_id",
        StringType(),
        True
    ),
    StructField(
        "type",
        StringType(),
        True
    ),
    StructField(
        "amount",
        DoubleType(),
        True
    ),
    StructField(
        "oldbalanceOrg",
        DoubleType(),
        True
    ),
    StructField(
        "newbalanceOrig",
        DoubleType(),
        True
    ),
    StructField(
        "oldbalanceDest",
        DoubleType(),
        True
    ),
    StructField(
        "newbalanceDest",
        DoubleType(),
        True
    )
])


# ============================================================
# 5. READ TRANSACTIONS FROM KAFKA
# ============================================================

messages = kafka_df.selectExpr(
    "CAST(value AS STRING) AS transaction"
)


# ============================================================
# 6. PARSE TRANSACTION JSON
# ============================================================

transactions = messages.select(
    from_json(
        col("transaction"),
        transaction_schema
    ).alias("data")
).select(
    "data.*"
)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

features = transactions \
    .withColumn(
        "isPayment",
        when(
            col("type") == "PAYMENT",
            1
        ).otherwise(0)
    ) \
    .withColumn(
        "isTransfer",
        when(
            col("type") == "TRANSFER",
            1
        ).otherwise(0)
    ) \
    .withColumn(
        "isCashOut",
        when(
            col("type") == "CASH_OUT",
            1
        ).otherwise(0)
    ) \
    .withColumn(
        "isDebit",
        when(
            col("type") == "DEBIT",
            1
        ).otherwise(0)
    ) \
    .withColumn(
        "isCashIn",
        when(
            col("type") == "CASH_IN",
            1
        ).otherwise(0)
    )


# ============================================================
# 8. CREATE BALANCE FEATURES
# ============================================================

features = features \
    .withColumn(
        "origBalanceDiff",
        col("oldbalanceOrg") -
        col("newbalanceOrig")
    ) \
    .withColumn(
        "destBalanceDiff",
        col("newbalanceDest") -
        col("oldbalanceDest")
    )


# ============================================================
# 9. ML PREDICTION
# ============================================================

predictions = model.transform(
    features
)


# ============================================================
# 10. EXTRACT FRAUD PROBABILITY
# ============================================================

predictions = predictions.withColumn(
    "fraudProbability",
    vector_to_array(
        col("probability")
    )[1]
)


# ============================================================
# 11. APPLY ML DECISION THRESHOLD
# ============================================================

predictions = predictions.withColumn(
    "result",
    when(
        col("fraudProbability") >= 0.99,
        "FRAUD"
    ).otherwise(
        "GENUINE"
    )
)


# ============================================================
# 12. PREPARE RESULT
# ============================================================

results = predictions.select(
    col("transaction_id"),
    col("fraudProbability"),
    col("result")
)


# ============================================================
# 13. CONVERT RESULT TO JSON
# ============================================================

kafka_results = results.select(
    to_json(
        struct(
            col("transaction_id"),
            col("fraudProbability"),
            col("result")
        )
    ).alias("value")
)


# ============================================================
# 14. SEND RESULT TO KAFKA
# ============================================================

query = kafka_results.writeStream \
    .format("kafka") \
    .option(
        "kafka.bootstrap.servers",
        "localhost:9092"
    ) \
    .option(
        "topic",
        "fraud_results"
    ) \
    .option(
        "checkpointLocation",
        "output/fraud_results_checkpoint"
    ) \
    .outputMode("append") \
    .start()

print(
    "Real-time fraud detection streaming is running!"
)


# ============================================================
# 15. KEEP STREAMING RUNNING
# ============================================================

query.awaitTermination()