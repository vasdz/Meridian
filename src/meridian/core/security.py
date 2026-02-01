"""Security utilities - encryption, hashing, secrets."""

import secrets
import hashlib
from typing import Optional

from cryptography.fernet import Fernet
from passlib.context import CryptContext

from meridian.core.config import settings


# Password hashing context - using argon2 (more modern and secure than bcrypt)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class Encryptor:
    """Symmetric encryption using Fernet."""

    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            # Derive key from secret_key
            key = self._derive_key(settings.secret_key)
        self._fernet = Fernet(key)

    def _derive_key(self, secret: str) -> bytes:
        """Derive Fernet key from secret."""
        import base64
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"meridian-encryption-salt",
            iterations=100000,
        )
        key = kdf.derive(secret.encode())
        return base64.urlsafe_b64encode(key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext string."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext string."""
        return self._fernet.decrypt(ciphertext.encode()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash password for storage."""
    return pwd_context.hash(password)


def generate_api_key(prefix: str = "mk") -> str:
    """Generate a secure API key."""
    # Format: prefix_randomstring
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"


def hash_api_key(api_key: str) -> str:
    """Hash API key for storage (one-way)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return secrets.compare_digest(a, b)


def generate_token(length: int = 32) -> str:
    """Generate a random token."""
    return secrets.token_urlsafe(length)

