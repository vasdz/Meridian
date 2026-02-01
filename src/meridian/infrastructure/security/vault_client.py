"""HashiCorp Vault client for secrets management."""

from meridian.core.config import settings
from meridian.core.logging import get_logger

logger = get_logger(__name__)


class VaultClient:
    """HashiCorp Vault client."""

    def __init__(
        self,
        address: str | None = None,
        token: str | None = None,
    ):
        self.address = address or settings.vault_address
        self.token = token or settings.vault_token
        self._client = None

    def _get_client(self):
        """Get Vault client (lazy initialization)."""
        if self._client is None and settings.vault_enabled:
            try:
                import hvac

                self._client = hvac.Client(
                    url=self.address,
                    token=self.token,
                )

                if self._client.is_authenticated():
                    logger.info("Vault client authenticated")
                else:
                    logger.error("Vault authentication failed")
                    self._client = None

            except ImportError:
                logger.warning("hvac not installed")

        return self._client

    def get_secret(self, path: str, key: str | None = None) -> str | None:
        """Get secret from Vault."""
        client = self._get_client()

        if client is None:
            logger.debug("Vault disabled, returning None")
            return None

        try:
            secret = client.secrets.kv.v2.read_secret_version(path=path)
            data = secret["data"]["data"]

            if key:
                return data.get(key)
            return data

        except Exception as e:
            logger.error("Failed to read secret", path=path, error=str(e))
            return None

    def put_secret(self, path: str, data: dict) -> bool:
        """Store secret in Vault."""
        client = self._get_client()

        if client is None:
            return False

        try:
            client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
            )
            logger.info("Secret stored", path=path)
            return True

        except Exception as e:
            logger.error("Failed to store secret", path=path, error=str(e))
            return False

    def get_database_credentials(self, role: str = "meridian-db") -> dict | None:
        """Get dynamic database credentials."""
        client = self._get_client()

        if client is None:
            return None

        try:
            creds = client.secrets.database.generate_credentials(role)
            return {
                "username": creds["data"]["username"],
                "password": creds["data"]["password"],
            }
        except Exception as e:
            logger.error("Failed to get DB credentials", error=str(e))
            return None


# Singleton
_vault_client: VaultClient | None = None


def get_vault_client() -> VaultClient:
    """Get Vault client instance."""
    global _vault_client

    if _vault_client is None:
        _vault_client = VaultClient()

    return _vault_client
