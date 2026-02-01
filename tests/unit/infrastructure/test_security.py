"""Unit tests for security infrastructure."""

import pytest

from meridian.infrastructure.security.encryption import FieldEncryptor
from meridian.core.security import (
    hash_password,
    verify_password,
    generate_api_key,
    hash_api_key,
    constant_time_compare,
)


class TestFieldEncryptor:
    """Tests for field-level encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encrypt/decrypt returns original value."""
        encryptor = FieldEncryptor()

        plaintext = "sensitive-data-123"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypt_produces_different_output(self):
        """Test that encryption is non-deterministic (unique nonce)."""
        encryptor = FieldEncryptor()

        plaintext = "sensitive-data-123"
        encrypted1 = encryptor.encrypt(plaintext)
        encrypted2 = encryptor.encrypt(plaintext)

        assert encrypted1 != encrypted2

    def test_encrypt_unicode(self):
        """Test encryption of unicode strings."""
        encryptor = FieldEncryptor()

        plaintext = "Привет мир! 🔐"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == plaintext

    def test_wrong_key_fails(self):
        """Test that wrong key fails to decrypt."""
        encryptor1 = FieldEncryptor()
        encryptor2 = FieldEncryptor()  # Different key

        encrypted = encryptor1.encrypt("secret")

        with pytest.raises(Exception):
            encryptor2.decrypt(encrypted)


class TestPasswordHashing:
    """Tests for password hashing."""

    def test_hash_and_verify(self):
        """Test password hashing and verification."""
        password = "SecurePassword123!"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_wrong_password_fails(self):
        """Test that wrong password fails verification."""
        password = "SecurePassword123!"
        hashed = hash_password(password)

        assert verify_password("WrongPassword", hashed) is False

    def test_hash_is_unique(self):
        """Test that hashes are unique (salted)."""
        password = "SecurePassword123!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2


class TestAPIKeyGeneration:
    """Tests for API key generation."""

    def test_generate_api_key_format(self):
        """Test API key format."""
        key = generate_api_key(prefix="mk")

        assert key.startswith("mk_")
        assert len(key) > 20

    def test_generate_api_key_unique(self):
        """Test that API keys are unique."""
        key1 = generate_api_key()
        key2 = generate_api_key()

        assert key1 != key2

    def test_hash_api_key(self):
        """Test API key hashing."""
        # Use dynamically generated key to avoid secret detection
        import secrets
        key = f"mk_test_{secrets.token_hex(12)}"
        hashed = hash_api_key(key)

        assert hashed != key
        assert len(hashed) == 64  # SHA-256 hex

    def test_constant_time_compare(self):
        """Test constant-time string comparison."""
        assert constant_time_compare("abc", "abc") is True
        assert constant_time_compare("abc", "def") is False
        assert constant_time_compare("abc", "ab") is False

