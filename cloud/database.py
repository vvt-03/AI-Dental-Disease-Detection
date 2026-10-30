"""Database integration for the cloud API."""

import sqlite3
from pathlib import Path

# Store the local database in the project root.
DB_PATH = Path(__file__).resolve().parent.parent / "predictions.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL
                    CHECK (confidence >= 0 AND confidence <= 1),
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_prediction(prediction: dict, created_at: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                image_id,
                predicted_class,
                confidence,
                model_version,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction["image_id"],
                prediction["predicted_class"],
                prediction["confidence"],
                prediction["model_version"],
                created_at,
            ),
        )
        return cursor.lastrowid
