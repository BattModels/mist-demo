from pathlib import Path
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").getOrCreate()

dataset_path = Path("realspace")
data_split = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

data = spark.read.text("REALSpace_t2")
for name, split in zip(data_split.keys(), data.randomSplit(data_split.values())):
    n = max(int(split.count() / 100000), 1)
    split.repartition(n).write.mode("overwrite").text(
        str(dataset_path.joinpath("data", name))
    )
