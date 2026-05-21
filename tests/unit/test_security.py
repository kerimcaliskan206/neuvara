import pytest

from app.core.security import hash_password, verify_password


def test_hash_password_returns_string():
    hashed = hash_password("SecurePass@1")
    assert isinstance(hashed, str)
    assert hashed != "SecurePass@1"


def test_verify_password_correct():
    hashed = hash_password("SecurePass@1")
    assert verify_password("SecurePass@1", hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("SecurePass@1")
    assert verify_password("WrongPass@1", hashed) is False


def test_hash_is_unique_per_call():
    h1 = hash_password("SecurePass@1")
    h2 = hash_password("SecurePass@1")
    assert h1 != h2  # bcrypt generates different salts each time
