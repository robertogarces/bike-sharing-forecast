import logging
import subprocess

logger = logging.getLogger(__name__)


def run_command(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """
    Run a shell command and raise if it fails.

    capture_output=False (default) streams the command's output straight to
    the console — used for long-running commands like `dvc repro` where you
    want to see progress live. capture_output=True captures stdout/stderr
    instead (as text) and returns them on the result, needed by callers that
    parse the output (e.g. `gh issue list --json`).
    """
    logger.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, capture_output=capture_output, text=capture_output)
