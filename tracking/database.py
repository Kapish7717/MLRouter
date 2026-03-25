# tracking/database.py
import sqlite3
from datetime import datetime
import os

DB_PATH = "tracking/predictions.db"

def init_db():
    os.makedirs("tracking", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Every prediction gets logged here
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            experiment_id   TEXT    NOT NULL,
            model_id        TEXT    NOT NULL,
            variant         TEXT    NOT NULL,   -- 'A' or 'B'
            input_hash      TEXT,               -- hash of input
            prediction      INTEGER NOT NULL,   -- 0 or 1
            confidence      REAL    NOT NULL,   -- probability
            latency_ms      REAL    NOT NULL,   -- inference time
            ground_truth    INTEGER             -- actual label if known
        )
    """)

    # Aggregated experiment stats (updated periodically)
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiment_stats (
            experiment_id   TEXT PRIMARY KEY,
            model_a_requests INTEGER DEFAULT 0,
            model_b_requests INTEGER DEFAULT 0,
            model_a_avg_conf REAL    DEFAULT 0,
            model_b_avg_conf REAL    DEFAULT 0,
            model_a_avg_lat  REAL    DEFAULT 0,
            model_b_avg_lat  REAL    DEFAULT 0,
            model_a_accuracy REAL    DEFAULT 0,
            model_b_accuracy REAL    DEFAULT 0,
            last_updated     TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialised")

def log_prediction(experiment_id: str, model_id: str,
                   variant: str, prediction: int,
                   confidence: float, latency_ms: float,
                   input_hash: str = None):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (timestamp, experiment_id, model_id, variant,
         input_hash, prediction, confidence, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        experiment_id, model_id, variant,
        input_hash, prediction, confidence, latency_ms
    ))
    conn.commit()
    conn.close()

def get_experiment_metrics(experiment_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        SELECT
            variant,
            COUNT(*)                    AS requests,
            AVG(confidence)             AS avg_confidence,
            AVG(latency_ms)             AS avg_latency,
            SUM(CASE WHEN prediction=1
                THEN 1 ELSE 0 END) * 1.0
                / COUNT(*)              AS positive_rate
        FROM predictions
        WHERE experiment_id = ?
        GROUP BY variant
    """, (experiment_id,))

    rows = c.fetchall()
    conn.close()

    metrics = {}
    for row in rows:
        variant, requests, avg_conf, avg_lat, pos_rate = row
        metrics[f"model_{variant.lower()}"] = {
            "variant":         variant,
            "total_requests":  requests,
            "avg_confidence":  round(avg_conf or 0, 4),
            "avg_latency_ms":  round(avg_lat or 0, 2),
            "positive_rate":   round(pos_rate or 0, 4),
        }

    return metrics

def get_recent_predictions(
    experiment_id: str, limit: int = 50
) -> list:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT timestamp, model_id, variant,
               prediction, confidence, latency_ms
        FROM predictions
        WHERE experiment_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (experiment_id, limit))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "timestamp":   r[0],
            "model_id":    r[1],
            "variant":     r[2],
            "prediction":  r[3],
            "confidence":  r[4],
            "latency_ms":  r[5],
        }
        for r in rows
    ]

init_db()