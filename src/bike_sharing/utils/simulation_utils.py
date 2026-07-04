import json
from pathlib import Path


def load_simulation_state(state_path: Path) -> dict:
    """
    Load the simulation state from disk.
    Raises if the simulation has not been initialized.
    """
    if not state_path.exists():
        raise RuntimeError(
            "No simulation state found. Run shift_dates.py first to initialize the simulation."
        )
    with open(state_path) as f:
        return json.load(f)
