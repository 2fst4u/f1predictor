from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

from .util import get_logger

logger = get_logger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

def get_engine(db_path: str):
    """Creates a SQLAlchemy engine connected to the SQLite database."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    sqlite_url = f"sqlite:///{db_path}"
    # check_same_thread=False is needed for SQLite in FastAPI since different threads
    # might use the same connection.
    engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    return engine

def get_session_local(engine):
    """Creates a session factory bound to the engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(engine):
    """Creates all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")

# Dependency to use in FastAPI routes
def get_db(session_factory):
    """Dependency that yields a database session."""
    def _get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()
    return _get_db