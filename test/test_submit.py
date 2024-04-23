import pytest
from submit.submit import cli
from typer.testing import CliRunner

runner = CliRunner()


def test_polaris_default():
    result = runner.invoke(cli, ["submit/polaris.j2"])
    assert result.exit_code == 0


def test_data_explicit():
    result = runner.invoke(
        cli, ["submit/polaris.j2", "--no-default", "--data", "submit/pretrain.yaml"]
    )
    assert result.exit_code == 0


def test_multiple_data():

    result = runner.invoke(
        cli,
        [
            "submit/polaris.j2",
            "--data",
            "submit/pretrain.yaml",
            "--data",
            "submit/nsys.yaml",
        ],
    )
    assert result.exit_code == 0

@pytest.fixture
def finetuning_configs():
    return [
        "submit/finetune.yaml",
        "submit/multitask_classification.yaml",
    ]

def test_finetuning(finetuning_configs):
    for config in finetuning_configs:
        result = runner.invoke(
            cli,
            [
                "submit/h001.j2",
                "--data",
                config
            ],
        )
        assert result.exit_code == 0