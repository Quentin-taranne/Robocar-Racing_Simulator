import csv
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np


class DataLogger:
    """Stream small driving datasets to CSV for supervised training."""

    def __init__(self, output_dir: str = "data", agent_id: int = 0) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filepath = output_dir / f"drive_{agent_id}_{timestamp}.csv"
        self._file = self.filepath.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "timestamp",
                "episode",
                "step",
                "agent_id",
                "reward",
                "done",
                "action",
                "obs",
            ],
        )
        self._writer.writeheader()

    def log(
        self,
        *,
        episode: int,
        step: int,
        agent_id: int,
        reward: float,
        done: bool,
        action: Iterable[float],
        obs: Iterable[float],
    ) -> None:
        self._writer.writerow(
            {
                "timestamp": time.time(),
                "episode": episode,
                "step": step,
                "agent_id": agent_id,
                "reward": reward,
                "done": int(done),
                "action": json.dumps(np.asarray(action, dtype=float).tolist()),
                "obs": json.dumps(np.asarray(obs, dtype=float).tolist()),
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
