from typer.testing import CliRunner
from pathlib import Path
from classic_example.cli import app

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["load-cola-data", "/tmp/data"])
    assert result.exit_code == 0, result.exception
    assert Path("/tmp/data/train.csv").exists()
    assert Path("/tmp/data/val.csv").exists()
    assert Path("/tmp/data/test.csv").exists()

    result = runner.invoke(app, ["train", "tests/data/test_config.json"])
    assert result.exit_code == 0, result.exception
    assert Path("/tmp/results").exists()

    result = runner.invoke(app, ["upload-to-registry", "cli-test", "/tmp/results"])
    assert result.exit_code == 0, result.exception

