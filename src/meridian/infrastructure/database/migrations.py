"""Database migrations helpers."""

import asyncio

from alembic import command
from alembic.config import Config

from meridian.core.config import settings


def run_migrations() -> None:
    """Run database migrations."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", settings.sync_database_url)
    command.upgrade(alembic_cfg, "head")


def create_migration(message: str) -> None:
    """Create a new migration."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", settings.sync_database_url)
    command.revision(alembic_cfg, autogenerate=True, message=message)

