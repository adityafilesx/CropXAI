"""
CropXAI Experiment Provenance Database
=======================================
A SQLite-backed store for ALL experimental results.
This is the SINGLE SOURCE OF TRUTH for the manuscript.

RULE: No figure or table may contain a number that does not
      originate from a row in this database.
      The build will FAIL if provenance cannot be verified.
"""

import sqlite3
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = Path("results/experiment_db.sqlite")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS experiments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          TEXT NOT NULL UNIQUE,
        timestamp       REAL NOT NULL,
        model           TEXT NOT NULL,
        dataset         TEXT NOT NULL,
        seed            INTEGER NOT NULL,
        epochs          INTEGER NOT NULL,
        finetune_mode   TEXT NOT NULL,
        git_sha         TEXT,
        hardware        TEXT
    );

    CREATE TABLE IF NOT EXISTS metrics (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          TEXT NOT NULL REFERENCES experiments(run_id),
        metric_name     TEXT NOT NULL,
        metric_value    REAL NOT NULL,
        is_validated    INTEGER NOT NULL DEFAULT 0,
        UNIQUE(run_id, metric_name)
    );

    CREATE TABLE IF NOT EXISTS figures (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        figure_name     TEXT NOT NULL UNIQUE,
        source_run_ids  TEXT NOT NULL,   -- JSON array of run_ids
        generated_at    REAL NOT NULL
    );
    """)
    conn.commit()


def store_run(
    model: str,
    dataset: str,
    seed: int,
    epochs: int,
    finetune_mode: str,
    metrics: Dict[str, float],
    git_sha: str = "unknown",
    hardware: str = "unknown"
) -> str:
    """
    Stores a single completed experiment run with all its metrics.
    Returns the unique run_id.
    """
    fingerprint = f"{model}|{dataset}|{seed}|{epochs}|{finetune_mode}|{git_sha}"
    run_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO experiments
               (run_id, timestamp, model, dataset, seed, epochs, finetune_mode, git_sha, hardware)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run_id, time.time(), model, dataset, seed, epochs, finetune_mode, git_sha, hardware)
        )
        for name, value in metrics.items():
            conn.execute(
                """INSERT OR REPLACE INTO metrics (run_id, metric_name, metric_value, is_validated)
                   VALUES (?,?,?,1)""",
                (run_id, name, value)
            )
        conn.commit()
        logger.info(f"Stored run [{run_id}]: {model} on {dataset} (seed={seed}) — {len(metrics)} metrics")
        return run_id
    finally:
        conn.close()


def register_figure(figure_name: str, source_run_ids: list):
    """Records which run_ids were used to generate a given figure."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO figures (figure_name, source_run_ids, generated_at)
               VALUES (?,?,?)""",
            (figure_name, json.dumps(source_run_ids), time.time())
        )
        conn.commit()
    finally:
        conn.close()


def query_metric(model: str, dataset: str, metric_name: str) -> Optional[Dict[str, float]]:
    """
    Retrieves a metric's mean, std, and count across all seeds for a model/dataset.
    Returns None if no data exists — triggering a build failure.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT m.metric_value
               FROM metrics m
               JOIN experiments e ON m.run_id = e.run_id
               WHERE e.model=? AND e.dataset=? AND m.metric_name=? AND m.is_validated=1""",
            (model, dataset, metric_name)
        ).fetchall()

        if not rows:
            return None

        import numpy as np
        vals = [r[0] for r in rows]
        n = len(vals)
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        # 95% bootstrap CI
        if n > 1:
            boots = [np.mean(np.random.choice(vals, size=n, replace=True)) for _ in range(2000)]
            ci_lo = float(np.percentile(boots, 2.5))
            ci_hi = float(np.percentile(boots, 97.5))
        else:
            ci_lo = ci_hi = mean

        return {"mean": mean, "std": std, "n": n, "ci_lo": ci_lo, "ci_hi": ci_hi}
    finally:
        conn.close()


def list_all_runs() -> list:
    """Returns all stored runs as dicts."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


class ProvenanceGuard:
    """
    Build-time provenance checker.
    Raises BuildError if any figure/table references a metric
    not found in the experiment database.
    """

    class BuildError(Exception):
        pass

    def __init__(self, strict: bool = True):
        self.strict = strict
        self._missing = []

    def require(self, model: str, dataset: str, metric: str) -> float:
        """
        Returns the mean value of a metric from real experiments.
        If strict=True, raises BuildError when data is missing.
        If strict=False, logs a warning and returns NaN.
        """
        result = query_metric(model, dataset, metric)
        if result is None:
            msg = f"PROVENANCE FAILURE: No data for {model}/{dataset}/{metric}"
            self._missing.append(msg)
            if self.strict:
                raise self.BuildError(msg)
            else:
                logger.warning(msg)
                import math
                return math.nan
        return result["mean"]

    def summary(self) -> str:
        if not self._missing:
            return "✓ All figures have experiment provenance."
        return f"✗ {len(self._missing)} MISSING provenance entries:\n" + "\n".join(self._missing)
