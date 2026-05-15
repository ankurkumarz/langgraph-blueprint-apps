"""
Entrypoint — imports the FastAPI app from app.app and runs uvicorn.

Usage:
  python main.py              # development (auto-reload)
  uvicorn app.app:app --host …    # production
"""

import os

from app.app import app  # noqa: F401  — re-export for `uvicorn main:app`

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("ENV", "development") == "development",
    )
