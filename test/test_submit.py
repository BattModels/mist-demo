import pytest
from pathlib import Path
from submit.submit import cli
from typer.testing import CliRunner

runner = CliRunner()


def get_scripts():
    submit_dir = Path(__file__).parent.parent.joinpath("submit")
    for file in submit_dir.iterdir():
        if file.suffix == ".j2" and file.name != "cliopts.j2":
            yield file.relative_to(submit_dir.parent)


@pytest.fixture(scope="module", params=get_scripts())
def script(request):
    return request.param


def get_data():
    submit_dir = Path(__file__).parent.parent.joinpath("submit")
    for file in submit_dir.iterdir():
        if file.suffix in [".yaml", ".yml"]:
            yield file


@pytest.fixture(scope="module", params=get_data())
def data(request):
    return request.param


def test_polaris_default(script):
    result = runner.invoke(cli, [str(script)])
    assert result.exit_code == 0


def test_data_explicit(script: Path, data: Path):
    result = runner.invoke(cli, [str(script), "--no-default", "--data", str(data)])
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
