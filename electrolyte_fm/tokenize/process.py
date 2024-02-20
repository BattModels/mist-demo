import typer
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pathlib import Path

cli = typer.Typer()

spark = SparkSession(SparkContext())

data_split = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


@cli.command()
def split_data(path: Path, dataset_path: Path):
    data = spark.read.text(str(path.joinpath("*.txt")))
    splits = data.randomSplit([x for x in data_split.values()])
    for name, split in zip(data_split.keys(), splits):
        split.write.mode("overwrite").text(str(dataset_path.joinpath("data", name)))


if __name__ == "__main__":
    cli()
