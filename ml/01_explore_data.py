from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("FraudDetectionDataExploration") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "data/paysim.csv",
    header=True,
    inferSchema=True
)

# Display number of rows and columns
print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

# Display column names
print("\nColumns:")
print(df.columns)

# Display schema
print("\nSchema:")
df.printSchema()

# Display first 10 rows
print("\nFirst 10 rows:")
df.show(10, truncate=False)

# Stop Spark
spark.stop()