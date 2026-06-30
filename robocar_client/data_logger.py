import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np


class DataLogger:
    """Stream small driving datasets to CSV for supervised training.

    The last `trim_seconds` of every session are held back in memory and
    never written to disk: when you stop recording (crash, deliberate
    interrupt, or just releasing the controls), those last frames are almost
    always the bad moment itself, not a clean driving decision. Buffering and
    dropping them on close is cheap and avoids re-reading/rewriting the whole
    CSV at the end of a long session.
    """

    def __init__(self, output_dir: str = "data", agent_id: int = 0, trim_seconds: float = 2.0) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filepath = output_dir / f"drive_{agent_id}_{timestamp}.csv"
        self.trim_seconds = trim_seconds
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
        self._pending: deque[tuple[float, dict]] = deque()

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
        now = time.time()
        row = {
            "timestamp": now,
            "episode": episode,
            "step": step,
            "agent_id": agent_id,
            "reward": reward,
            "done": int(done),
            "action": json.dumps(np.asarray(action, dtype=float).tolist()),
            "obs": json.dumps(np.asarray(obs, dtype=float).tolist()),
        }
        self._pending.append((now, row))
        self._flush_old(now)

    def _flush_old(self, now: float) -> None:
        cutoff = now - self.trim_seconds
        while self._pending and self._pending[0][0] <= cutoff:
            _, row = self._pending.popleft()
            self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        # Les `trim_seconds` les plus recentes restent dans `_pending` et ne
        # sont jamais ecrites : c'est volontaire, voir docstring de la classe.
        dropped = len(self._pending)
        self._pending.clear()
        self._file.close()
        if dropped:
            print(f"[DataLogger] {dropped} frames des {self.trim_seconds:.1f}s avant l'arret ne sont pas enregistrees ({self.filepath.name}).")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
