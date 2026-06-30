from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pathlib import Path
import os
 
BASE_DIR = Path(__file__).resolve().parent # folder your notebook/script is currently running in
mock_csv = "mock.csv"

csv_path = (BASE_DIR / mock_csv).resolve()




spark = SparkSession.builder.appName("Mock").getOrCreate()
df = spark.read.csv(str(csv_path), header=True, inferSchema=True)

assembler = VectorAssembler(inputCols=[
    "attendance_percent",
    "assignment_submitted",
    "assignment_score",
    "quiz_score",
    "lms_logins",
    "helpdesk_requests",
    "last_login_days_ago"], outputCol="features")

indexer = StringIndexer(inputCol="risk_level", outputCol="label")

output = assembler.transform(df)
output = indexer.fit(output).transform(output)
df_final=output.select("features", "label")

train, test = df_final.randomSplit([0.7, 0.3], seed=42)

lr = LogisticRegression(labelCol='label')

lrm = lr.fit(train)
pred_labels = lrm.transform(test)

accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(pred_labels)

precision = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(pred_labels)

recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(pred_labels)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")