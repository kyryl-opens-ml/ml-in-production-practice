from pathlib import Path

from classic_example.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["load-sst2-data", "/tmp/data"])
    assert result.exit_code == 0, result.exception
    assert Path("/tmp/data/train.csv").exists()
    assert Path("/tmp/data/val.csv").exists()
    assert Path("/tmp/data/test.csv").exists()

    result = runner.invoke(app, ["train", "tests/data/test_config.json"])
    assert result.exit_code == 0, result.exception
    assert Path("/tmp/results").exists()

    result = runner.invoke(app, ["upload-to-registry", "cli-test", "/tmp/results"])
    assert result.exit_code == 0, result.exception
