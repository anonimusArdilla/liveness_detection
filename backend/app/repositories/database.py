"""Database engine and session management supporting SQLite and PostgreSQL."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.models.session import Base


class DatabaseManager:
    """Manages SQLAlchemy engine and session lifecycle."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        connect_args: dict[str, Any] = {}

        if database_url.startswith("sqlite:///"):
            # Ensure parent directory exists for file-backed SQLite databases
            db_path_str = database_url.replace("sqlite:///", "", 1)
            if db_path_str and db_path_str != ":memory:":
                db_path = Path(db_path_str)
                db_path.parent.mkdir(parents=True, exist_ok=True)
            connect_args["check_same_thread"] = False

        self.engine: Engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    def init_schema(self) -> None:
        """Create tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    def check_health(self) -> bool:
        """Execute a lightweight query to verify database connectivity."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_session(self) -> Generator[Session, None, None]:
        """Yield a transactional database session."""
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()


_db_manager: DatabaseManager | None = None


def get_db_manager(settings: Settings | None = None) -> DatabaseManager:
    """Get or initialize singleton DatabaseManager."""
    global _db_manager
    resolved_settings = settings or get_settings()
    if _db_manager is None or _db_manager.database_url != resolved_settings.database_url:
        _db_manager = DatabaseManager(resolved_settings.database_url)
        _db_manager.init_schema()
    return _db_manager
