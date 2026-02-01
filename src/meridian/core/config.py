"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings following 12-factor app principles."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Meridian", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Security
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")
    encryption_salt: str = Field(
        default="change-me-in-production-must-be-16-bytes",
        alias="ENCRYPTION_SALT",
        description="Salt for encryption key derivation (min 16 characters)",
    )
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30)
    api_key_header: str = Field(default="X-API-Key")

    # CORS
    cors_origins: list[str] = Field(default=["*"])

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # nosec B104 - configurable via env
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=4, alias="WORKERS")

    # Database
    db_driver: str = Field(default="sqlite+aiosqlite", alias="DB_DRIVER")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="meridian.db", alias="DB_NAME")
    db_user: str = Field(default="meridian", alias="DB_USER")
    db_password: str = Field(default="meridian", alias="DB_PASSWORD")
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")

    @property
    def database_url(self) -> str:
        """Build database URL."""
        if "sqlite" in self.db_driver:
            return f"{self.db_driver}:///./{self.db_name}"
        return (
            f"{self.db_driver}://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sync_database_url(self) -> str:
        """Build sync database URL for Alembic."""
        if "sqlite" in self.db_driver:
            return f"sqlite:///./{self.db_name}"
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # Redis
    redis_enabled: bool = Field(default=False, alias="REDIS_ENABLED")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")

    @property
    def redis_url(self) -> str:
        """Build Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=60)
    rate_limit_burst_size: int = Field(default=100)

    # Vault
    vault_enabled: bool = Field(default=False, alias="VAULT_ENABLED")
    vault_address: str = Field(default="http://localhost:8200", alias="VAULT_ADDRESS")
    vault_token: str | None = Field(default=None, alias="VAULT_TOKEN")

    # Kafka
    kafka_enabled: bool = Field(default=False, alias="KAFKA_ENABLED")
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        alias="KAFKA_BOOTSTRAP_SERVERS",
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="meridian",
        alias="MLFLOW_EXPERIMENT_NAME",
    )

    # Spark
    spark_master: str = Field(default="local[*]", alias="SPARK_MASTER")

    # Celery
    celery_broker: str = Field(
        default="redis://localhost:6379/1",
        alias="CELERY_BROKER_URL",
    )
    celery_backend: str = Field(
        default="redis://localhost:6379/2",
        alias="CELERY_RESULT_BACKEND",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
