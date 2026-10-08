from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("FraudDetectionModelTraining") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Spark ML training started!")


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

df = spark.read.parquet(
    "output/processed_data"
)

print("\nProcessed data loaded successfully.")
print("Total rows:", df.count())


# ============================================================
# 3. CREATE LABEL
# ============================================================

df = df.withColumn(
    "label",
    col("isFraud").cast("double")
)


# ============================================================
# 4. DEFINE ML FEATURES
# ============================================================

feature_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isPayment",
    "isTransfer",
    "isCashOut",
    "isDebit",
    "isCashIn",
    "origBalanceDiff",
    "destBalanceDiff"
]

print("\nFeatures used for training:")

for feature in feature_columns:
    print("-", feature)


# ============================================================
# 5. CREATE FEATURE VECTOR
# ============================================================

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="rawFeatures"
)


# ============================================================
# 6. SCALE FEATURES
# ============================================================

scaler = StandardScaler(
    inputCol="rawFeatures",
    outputCol="features",
    withStd=True,
    withMean=False
)


# ============================================================
# 7. HANDLE CLASS IMBALANCE
# ============================================================

fraud_count = df.filter(
    col("label") == 1.0
).count()

genuine_count = df.filter(
    col("label") == 0.0
).count()

print("\nClass distribution:")
print("Genuine transactions:", genuine_count)
print("Fraud transactions:", fraud_count)

fraud_weight = genuine_count / fraud_count

print("\nFraud class weight:", fraud_weight)

df = df.withColumn(
    "classWeight",
    when(
        col("label") == 1.0,
        fraud_weight
    ).otherwise(1.0)
)


# ============================================================
# 8. SPLIT DATA
# ============================================================

train_data, validation_data, test_data = df.randomSplit(
    [0.70, 0.15, 0.15],
    seed=42
)

print("\nDataset split completed.")
print("Training rows:", train_data.count())
print("Validation rows:", validation_data.count())
print("Test rows:", test_data.count())


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=100,
    maxDepth=10,
    seed=42
)


# ============================================================
# 10. CREATE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=[
        assembler,
        scaler,
        rf
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest ML model...")

model = pipeline.fit(
    train_data
)

print("Model training completed.")


# ============================================================
# 12. VALIDATION PREDICTIONS
# ============================================================

validation_predictions = model.transform(
    validation_data
)

print("\nValidation predictions generated.")


# ============================================================
# 13. VALIDATION ROC-AUC
# ============================================================

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

validation_roc_auc = roc_evaluator.evaluate(
    validation_predictions
)


# ============================================================
# 14. VALIDATION PR-AUC
# ============================================================

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

validation_pr_auc = pr_evaluator.evaluate(
    validation_predictions
)

print("\nValidation Metrics")
print("------------------")
print("ROC-AUC:", validation_roc_auc)
print("PR-AUC :", validation_pr_auc)


# ============================================================
# 15. FRAUD PROBABILITY THRESHOLD ANALYSIS
# ============================================================

fraud_probability = udf(
    lambda probability: float(probability[1]),
    DoubleType()
)

validation_threshold_data = validation_predictions.withColumn(
    "fraudProbability",
    fraud_probability(col("probability"))
)

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99
]

print("\nThreshold Analysis")
print("------------------")

for threshold in thresholds:

    threshold_predictions = validation_threshold_data.withColumn(
        "thresholdPrediction",
        when(
            col("fraudProbability") >= lit(threshold),
            1.0
        ).otherwise(0.0)
    )

    counts = threshold_predictions.groupBy(
        "label",
        "thresholdPrediction"
    ).count().collect()

    tp = 0
    fp = 0
    fn = 0

    for row in counts:

        label = row["label"]
        prediction = row["thresholdPrediction"]
        count = row["count"]

        if label == 1.0 and prediction == 1.0:
            tp = count

        elif label == 0.0 and prediction == 1.0:
            fp = count

        elif label == 1.0 and prediction == 0.0:
            fn = count

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )


# ============================================================
# 16. SAVE MODEL
# ============================================================

model.write().overwrite().save(
    "model/fraud_model"
)

print("\nFinal ML model saved successfully.")
print("Location: model/fraud_model")


# ============================================================
# 17. TEST DATA EVALUATION
# ============================================================

test_predictions = model.transform(
    test_data
)

test_roc_auc = roc_evaluator.evaluate(
    test_predictions
)

test_pr_auc = pr_evaluator.evaluate(
    test_predictions
)

print("\nTest Metrics")
print("------------")
print("ROC-AUC:", test_roc_auc)
print("PR-AUC :", test_pr_auc)


# ============================================================
# 18. DEFAULT TEST CONFUSION MATRIX
# ============================================================

print("\nDefault Test Confusion Matrix")
print("-----------------------------")

test_predictions.groupBy(
    "label",
    "prediction"
).count().orderBy(
    "label",
    "prediction"
).show()


# ============================================================
# 19. STOP SPARK
# ============================================================

spark.stop()

print("\nSpark ML training completed successfully.")
