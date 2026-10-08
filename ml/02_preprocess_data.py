from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("FraudDetectionPreprocessing") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = spark.read.csv(
    "data/paysim.csv",
    header=True,
    inferSchema=True
)

print("Original rows:", df.count())

# ============================================================
# 3. REMOVE IDENTIFIER COLUMNS
# ============================================================
# nameOrig and nameDest identify individual accounts.
# They are not used directly as ML features.

df = df.drop(
    "nameOrig",
    "nameDest"
)

print("\nColumns after removing identifiers:")
print(df.columns)

# ============================================================
# 4. CHECK FOR NULL VALUES
# ============================================================

print("\nNull values:")

for column in df.columns:
    null_count = df.filter(
        col(column).isNull()
    ).count()

    print(column, ":", null_count)

# ============================================================
# 5. CONVERT TRANSACTION TYPE INTO NUMERIC FEATURES
# ============================================================

df = df.withColumn(
    "isPayment",
    when(col("type") == "PAYMENT", 1).otherwise(0)
)

df = df.withColumn(
    "isTransfer",
    when(col("type") == "TRANSFER", 1).otherwise(0)
)

df = df.withColumn(
    "isCashOut",
    when(col("type") == "CASH_OUT", 1).otherwise(0)
)

df = df.withColumn(
    "isDebit",
    when(col("type") == "DEBIT", 1).otherwise(0)
)

df = df.withColumn(
    "isCashIn",
    when(col("type") == "CASH_IN", 1).otherwise(0)
)

# Original categorical column is no longer required
df = df.drop("type")

# ============================================================
# 6. CREATE BALANCE DIFFERENCE FEATURES
# ============================================================

df = df.withColumn(
    "origBalanceDiff",
    col("oldbalanceOrg") - col("newbalanceOrig")
)

df = df.withColumn(
    "destBalanceDiff",
    col("newbalanceDest") - col("oldbalanceDest")
)

# ============================================================
# 7. DISPLAY FINAL FEATURES
# ============================================================

print("\nFinal columns:")
print(df.columns)

print("\nFinal schema:")
df.printSchema()

print("\nSample processed data:")
df.show(10, truncate=False)

# ============================================================
# 8. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nFraud distribution:")

df.groupBy("isFraud") \
    .count() \
    .show()

# ============================================================
# 9. SAVE PROCESSED DATA
# ============================================================

df.write \
    .mode("overwrite") \
    .parquet("output/processed_data")

print("\nProcessed data saved to:")
print("output/processed_data")

# ============================================================
# 10. STOP SPARK
# ============================================================

spark.stop()