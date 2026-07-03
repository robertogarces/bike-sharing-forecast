import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile

from bike_sharing.data.update_simulation import move_revealed_records


@pytest.fixture
def simulation_files():
    """
    Create temporary past and future CSV files for testing.
    Returns paths to both files inside a temp directory.
    """
    past_data = pd.DataFrame(
        {
            "dteday": ["2026-06-01"] * 24,
            "hr": list(range(24)),
            "cnt": list(range(100, 124)),
            "registered": list(range(80, 104)),
            "casual": list(range(20, 44)),
            "workingday": [1] * 24,
            "season": [2] * 24,
            "temp": [0.5] * 24,
            "hum": [0.6] * 24,
            "weathersit": [1] * 24,
            "windspeed": [0.2] * 24,
        }
    )

    future_data = pd.DataFrame(
        {
            "dteday": ["2026-06-02"] * 24,
            "hr": list(range(24)),
            "cnt": list(range(200, 224)),
            "registered": list(range(160, 184)),
            "casual": list(range(40, 64)),
            "workingday": [1] * 24,
            "season": [2] * 24,
            "temp": [0.5] * 24,
            "hum": [0.6] * 24,
            "weathersit": [1] * 24,
            "windspeed": [0.2] * 24,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        past_path = Path(tmpdir) / "hour_past.csv"
        future_path = Path(tmpdir) / "hour_future.csv"
        past_data.to_csv(past_path, index=False)
        future_data.to_csv(future_path, index=False)
        yield past_path, future_path


def test_move_revealed_records_moves_past_records(simulation_files):
    """
    Records whose datetime has already passed should be moved
    from hour_future.csv to hour_past.csv.
    """
    past_path, future_path = simulation_files

    # now is set to June 2 at noon — first 12 hours of June 2 should be revealed
    now = datetime(2026, 6, 2, 12, 0)
    updated_past, n_moved = move_revealed_records(past_path, future_path, now)

    assert n_moved == 13  # hours 0 to 12 inclusive


def test_move_revealed_records_does_not_move_future_records(simulation_files):
    """
    Records whose datetime has not yet passed should remain in hour_future.csv.
    """
    past_path, future_path = simulation_files

    now = datetime(2026, 6, 2, 12, 0)
    _, n_moved = move_revealed_records(past_path, future_path, now)

    remaining_future = pd.read_csv(future_path)
    assert len(remaining_future) == 24 - n_moved


def test_move_revealed_records_no_records_to_reveal(simulation_files):
    """
    If now is before all future records, nothing should be moved.
    """
    past_path, future_path = simulation_files

    # now is before June 2 — nothing should be revealed
    now = datetime(2026, 6, 1, 23, 0)
    _, n_moved = move_revealed_records(past_path, future_path, now)

    assert n_moved == 0
