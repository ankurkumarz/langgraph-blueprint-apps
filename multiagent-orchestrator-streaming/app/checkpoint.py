"""
SQLite-backed LangGraph checkpointer, message log, eval result store,
and reference trajectory store.

Provides:
  - get_checkpointer()             → SqliteSaver for LangGraph graph compilation
  - save_message()                 → persist a user query + session to the message log
  - update_message_response()      → fill in the assistant response once streaming ends
  - get_chat_history()             → recent sessions for /api/chat/history
  - search_chat_history()          → keyword search for /api/chat/search
  - save_eval_result()             → persist a trajectory or response eval score
  - get_eval_results()             → all eval results (newest first)
  - get_eval_results_by_session()  → eval results for one session
  - save_reference_trajectory()    → store a named "gold" message sequence for match evals
  - get_reference_trajectory()     → retrieve a stored reference by name
  - list_reference_trajectories()  → list all stored references
  - delete_reference_trajectory()  → remove a stored reference
"""

import json
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            eval_type   TEXT NOT NULL,
            score       REAL,
            reasoning   TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_session ON eval_results (session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_created ON eval_results (created_at)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reference_trajectories (
            name        TEXT PRIMARY KEY,
            description TEXT,
            messages_json TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
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


def save_eval_result(
    session_id: str,
    eval_type: str,
    score: Optional[float],
    reasoning: Optional[str],
) -> str:
    """Persist one evaluation result and return its generated id."""
    conn = _get_messages_conn()
    result_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    conn.execute(
        "INSERT INTO eval_results (id, session_id, eval_type, score, reasoning, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (result_id, session_id, eval_type, score, reasoning, now),
    )
    conn.commit()
    return result_id


def get_eval_results(limit: int = 100) -> list[dict]:
    """Return all eval results ordered by most recent first."""
    conn = _get_messages_conn()
    rows = conn.execute(
        """
        SELECT id, session_id, eval_type, score, reasoning, created_at
        FROM   eval_results
        ORDER  BY created_at DESC
        LIMIT  ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "eval_type": row[2],
            "score": row[3],
            "reasoning": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]


def get_eval_results_by_session(session_id: str) -> list[dict]:
    """Return eval results for a specific session."""
    conn = _get_messages_conn()
    rows = conn.execute(
        """
        SELECT id, session_id, eval_type, score, reasoning, created_at
        FROM   eval_results
        WHERE  session_id = ?
        ORDER  BY created_at DESC
        """,
        (session_id,),
    ).fetchall()
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "eval_type": row[2],
            "score": row[3],
            "reasoning": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]


# ── Reference Trajectories ────────────────────────────────────────────────────
#
# A reference trajectory is a named, serialised list of LangChain messages
# from a "gold standard" run.  Match evaluators compare a new run's messages
# against the stored reference to detect regressions.


def save_reference_trajectory(
    name: str,
    messages: list,
    description: str = "",
) -> None:
    """Persist a named reference trajectory (serialised messages).

    `messages` must be a list of LangChain BaseMessage objects or dicts
    already in `messages_to_dict` format.
    """
    from langchain_core.messages import BaseMessage, messages_to_dict

    if messages and isinstance(messages[0], BaseMessage):
        messages = messages_to_dict(messages)

    conn = _get_messages_conn()
    now = datetime.utcnow().isoformat() + "Z"
    conn.execute(
        """
        INSERT INTO reference_trajectories (name, description, messages_json, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description   = excluded.description,
            messages_json = excluded.messages_json,
            created_at    = excluded.created_at
        """,
        (name, description, json.dumps(messages), now),
    )
    conn.commit()


def get_reference_trajectory(name: str) -> Optional[dict]:
    """Return the stored reference trajectory for ``name``, or None if missing.

    The returned ``messages`` list contains deserialised LangChain BaseMessage
    objects ready for use with trajectory match evaluators.
    """
    from langchain_core.messages import messages_from_dict

    conn = _get_messages_conn()
    row = conn.execute(
        "SELECT name, description, messages_json, created_at"
        " FROM reference_trajectories WHERE name = ?",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return {
        "name": row[0],
        "description": row[1],
        "messages": messages_from_dict(json.loads(row[2])),
        "created_at": row[3],
    }


def list_reference_trajectories() -> list[dict]:
    """Return all stored references (without the raw message payloads)."""
    conn = _get_messages_conn()
    rows = conn.execute(
        "SELECT name, description, created_at"
        " FROM reference_trajectories ORDER BY created_at DESC"
    ).fetchall()
    return [
        {"name": row[0], "description": row[1], "created_at": row[2]}
        for row in rows
    ]


def delete_reference_trajectory(name: str) -> bool:
    """Delete a reference by name.  Returns True if a row was deleted."""
    conn = _get_messages_conn()
    cursor = conn.execute(
        "DELETE FROM reference_trajectories WHERE name = ?", (name,)
    )
    conn.commit()
    return cursor.rowcount > 0
