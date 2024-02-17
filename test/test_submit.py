from submit.submit import cli
from typer.testing import CliRunner

runner = CliRunner()


def test_polaris_default():
    result = runner.invoke(cli, ["submit/polaris.j2"])
    assert result.exit_code == 0


def test_polaris_explicit():
    result = runner.invoke(cli, ["submit/polaris.j2", "--data", "submit/polaris.yaml"])
    assert result.exit_code == 0
