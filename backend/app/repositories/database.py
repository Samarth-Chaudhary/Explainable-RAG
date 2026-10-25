"""Database setup and session management."""

from __future__ import annotations

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from backend.app.models.audit_models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(db_url: str | None = None) -> Engine:
    """Get or initialize SQLAlchemy database engine."""
    global _engine, _session_factory
    if _engine is None or db_url is not None:
        if db_url is None:
            db_path = os.getenv("DATABASE_URL", "sqlite:///data/audit_ledger.db")
        else:
            db_path = db_url

        if db_path.startswith("sqlite:///") and ":memory:" not in db_path:
            file_path = db_path.replace("sqlite:///", "")
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            db_path,
            connect_args={"check_same_thread": False} if db_path.startswith("sqlite") else {},
        )
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session(db_url: str | None = None) -> Session:
    """Provide a transactional database session."""
    engine = get_engine(db_url)
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _session_factory()


def init_db(engine: Engine | None = None) -> None:
    """Create all database tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(bind=engine)
