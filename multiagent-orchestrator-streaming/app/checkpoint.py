"""
SQLite-backed LangGraph checkpointer and message log.

Provides:
  - get_checkpointer()        → SqliteSaver for LangGraph graph compilation
  - save_message()            → persist a user query + session to the message log
  - update_message_response() → fill in the assistant response once streaming ends
  - get_chat_history()        → recent sessions for /api/chat/history
  - search_chat_history()     → keyword search for /api/chat/search
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

_DB_DIR = Path("data")
_CHECKPOINT_DB = _DB_DIR / "checkpoints.db"
_MESSAGES_DB = _DB_DIR / "messages.db"

_checkpointer: Optional[SqliteSaver] = None
_messages_conn: Optional[sqlite3.Connection] = None


def _init_messages_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            query       TEXT NOT NULL,
            response    TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_msg_session ON messages (session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_msg_created ON messages (created_at)"
    )
    conn.commit()


def get_checkpointer() -> SqliteSaver:
    """Return a singleton SqliteSaver backed by a local SQLite file.

    The connection is opened with check_same_thread=False so it can be
    shared across asyncio tasks (LangGraph serialises checkpoint writes).
    """
    global _checkpointer
    if _checkpointer is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_CHECKPOINT_DB), check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer


def _get_messages_conn() -> sqlite3.Connection:
    global _messages_conn
    if _messages_conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _messages_conn = sqlite3.connect(str(_MESSAGES_DB), check_same_thread=False)
        _init_messages_schema(_messages_conn)
    return _messages_conn


def save_message(session_id: str, query: str) -> str:
    """Insert a new user message row and return its generated id."""
    conn = _get_messages_conn()
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    conn.execute(
        "INSERT INTO messages (id, session_id, query, response, created_at)"
        " VALUES (?, ?, ?, NULL, ?)",
        (msg_id, session_id, query, now),
    )
    conn.commit()
    return msg_id


def update_message_response(msg_id: str, response: str) -> None:
    """Set the assistant response for a previously saved message row."""
    conn = _get_messages_conn()
    conn.execute(
        "UPDATE messages SET response = ? WHERE id = ?",
        (response, msg_id),
    )
    conn.commit()


def get_chat_history(limit: int = 50) -> list[dict]:
    """Return one summary row per session, ordered by most-recent activity."""
    conn = _get_messages_conn()
    rows = conn.execute(
        """
        SELECT session_id,
               query,
               MAX(created_at)  AS last_active,
               COUNT(*)         AS message_count
        FROM   messages
        GROUP  BY session_id
        ORDER  BY last_active DESC
        LIMIT  ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "id": row[0],
            "title": row[1][:80] if row[1] else "",
            "preview": row[1] if row[1] else "",
            "timestamp": row[2],
            "messageCount": row[3],
        }
        for row in rows
    ]


def search_chat_history(query: str, limit: int = 20) -> list[dict]:
    """Return messages whose query or response contains the search term."""
    conn = _get_messages_conn()
    needle = f"%{query.lower()}%"
    rows = conn.execute(
        """
        SELECT session_id, query, response, created_at
        FROM   messages
        WHERE  LOWER(query) LIKE ?
            OR LOWER(COALESCE(response, '')) LIKE ?
        ORDER  BY created_at DESC
        LIMIT  ?
        """,
        (needle, needle, limit),
    ).fetchall()
    return [
        {
            "id": row[0],
            "title": row[1][:80] if row[1] else "",
            "preview": row[1] if row[1] else "",
            "timestamp": row[3],
            "highlight": row[1],
        }
        for row in rows
    ]
