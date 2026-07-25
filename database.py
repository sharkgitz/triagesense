"""SQLite audit log + PII redaction (Section 6.3).

Every write goes through `redact_pii` first - the stored log never contains
raw member emails, phone numbers, or ID-like digit sequences.
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DB_PATH
from schema import RemediationResult

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
LONG_DIGIT_RE = re.compile(r"\d{6,}")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    request_type TEXT,
    urgency TEXT,
    confidence REAL,
    steps_executed TEXT,
    outputs TEXT,
    human_in_loop INTEGER,
    timestamp TEXT,
    raw_text_redacted TEXT
)
"""


def redact_pii(text: Optional[str]) -> str:
    if not text:
        return ""
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = LONG_DIGIT_RE.sub("[REDACTED_ID]", redacted)
    return redacted


def init_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def log_case(result: RemediationResult, raw_text: str = "", db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO cases
            (case_id, request_type, urgency, confidence, steps_executed, outputs,
             human_in_loop, timestamp, raw_text_redacted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.case_id,
                result.request_type.value,
                result.urgency.value,
                result.confidence,
                json.dumps(result.steps_executed),
                json.dumps(result.outputs),
                int(result.human_in_loop),
                result.timestamp,
                redact_pii(raw_text),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_cases(db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM cases", conn)
    finally:
        conn.close()

    if len(df) == 0:
        return df

    df["steps_executed"] = df["steps_executed"].apply(json.loads)
    df["outputs"] = df["outputs"].apply(json.loads)
    df["human_in_loop"] = df["human_in_loop"].astype(bool)
    return df
