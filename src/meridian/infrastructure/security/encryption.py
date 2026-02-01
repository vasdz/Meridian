"""Field-level encryption using AES-256-GCM."""

import base64
import os
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class FieldEncryptor:
    """AES-256-GCM encryption for sensitive fields."""

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize with encryption key.

        Args:
            key: 32-byte key for AES-256. If None, generates random key.
        """
        if key is None:
            key = AESGCM.generate_key(bit_length=256)

        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")

        self._aesgcm = AESGCM(key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string.

        Returns base64-encoded ciphertext with prepended nonce.
        """
        # Generate random nonce (96 bits = 12 bytes)
        nonce = os.urandom(12)

        # Encrypt
        ciphertext = self._aesgcm.encrypt(
            nonce,
            plaintext.encode("utf-8"),
            None,  # No associated data
        )

        # Combine nonce + ciphertext and encode
        combined = nonce + ciphertext
        return base64.b64encode(combined).decode("ascii")

    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt base64-encoded ciphertext.

        Expects nonce prepended to ciphertext.
        """
        # Decode
        combined = base64.b64decode(encrypted.encode("ascii"))

        # Split nonce and ciphertext
        nonce = combined[:12]
        ciphertext = combined[12:]

        # Decrypt
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")


# Singleton for common use
_default_encryptor: Optional[FieldEncryptor] = None


def get_field_encryptor() -> FieldEncryptor:
    """Get default field encryptor."""
    global _default_encryptor

    if _default_encryptor is None:
        # In production, load key from Vault
        from meridian.core.config import settings

        # Derive key from secret
        import hashlib
        key = hashlib.sha256(settings.secret_key.encode()).digest()

        _default_encryptor = FieldEncryptor(key)

    return _default_encryptor


def encrypt_pii(value: str) -> str:
    """Encrypt PII field."""
    return get_field_encryptor().encrypt(value)


def decrypt_pii(encrypted: str) -> str:
    """Decrypt PII field."""
    return get_field_encryptor().decrypt(encrypted)

