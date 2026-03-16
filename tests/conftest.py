"""Shared fixtures available to all test suites."""

import uuid

import pytest


@pytest.fixture
def other_user_id() -> uuid.UUID:
    """Return a second fixed UUID representing a different tenant.

    Used in multi-tenancy tests to verify that one user's data
    is never visible to another user's queries.

    Returns:
        A stable UUID distinct from user_id.
    """
    return uuid.UUID("99999999-9999-9999-9999-999999999999")


@pytest.fixture
def user_id() -> uuid.UUID:
    """Return a fixed UUID for deterministic test assertions.

    Returns:
        A stable UUID that can be used as a user identifier across all tests.
    """
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_texts() -> list[str]:
    """Return a small list of sample text documents.

    Returns:
        Three distinct strings representing documents to embed.
    """
    return ["first document", "second document", "third document"]


@pytest.fixture
def fake_vector() -> list[float]:
    """Return a short deterministic embedding vector for tests.

    The dimensionality is intentionally small (5) to keep tests lightweight.
    Unit tests verify call arguments, not embedding quality.

    Returns:
        A five-element float list used as a stub embedding vector.
    """
    return [0.1, 0.2, 0.3, 0.4, 0.5]
