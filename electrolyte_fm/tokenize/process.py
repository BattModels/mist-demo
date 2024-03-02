from datasets import Dataset
from torch.utils.data import random_split
import typer
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pathlib import Path
import numpy as np

cli = typer.Typer()


data_split = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def hashSplit(df, split, mod=10000):
    norm_weights = mod * np.cumsum(split) / np.sum(split)
    cuttoff = [w * 2 - mod for w in norm_weights]
    idf = df.withColumn("__hash", sf.xxhash64("value") % mod)
    splits = []
    low = -mod
    hash_col = sf.col("__hash")
    for high in cuttoff:
        split = idf.filter(hash_col >= low).filter(hash_col < high).drop(hash_col)
        splits.append(split)
        low = high
    return splits


@cli.command()
def split(path: Path, dataset_path: Path):
    spark = SparkSession(SparkContext())
    data = spark.read.text(str(path.joinpath("*.txt")))
    splits = hashSplit(data, [x for x in data_split.values()])
    for name, split in zip(data_split.keys(), splits):
        split.write.mode("overwrite").text(str(dataset_path.joinpath("data", name)))


@cli.command()
def shard(path: Path, dataset_path: Path):
    spark = SparkSession(SparkContext())
    data = spark.read.text(str(path.joinpath("*.txt")))
    data.write.mode("overwrite").text(str(dataset_path.joinpath("data", "raw")))


@cli.command()
def split_files(path: Path):
    files = [f for f in path.iterdir() if f.suffix.endswith(".txt")]
    for name, ds in zip(
        data_split.keys(), random_split(files, [x for x in data_split.values()])
    ):
        path.parent.joinpath(name).mkdir(exist_ok=True)
        for file in ds:
            dst = path.parent.joinpath(name, file.name)
            dst.unlink(missing_ok=True)
            dst.hardlink_to(file)


if __name__ == "__main__":
    cli()
