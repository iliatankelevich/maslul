"""Maslul's error hierarchy.

Everything Maslul raises is a :class:`MaslulError`. Provider drivers normalize their
SDK-specific failures into the :class:`ProviderError` subclasses, so the router's
resilience layer (M5) can act on them uniformly regardless of which backend failed.
"""

from __future__ import annotations


class MaslulError(Exception):
    """Base class for every error Maslul raises."""


class ConfigError(MaslulError):
    """Configuration is malformed — a bad model spec, an unknown provider/level, etc."""


class ProviderError(MaslulError):
    """A provider/SDK call failed. Base for the normalized provider failures below."""


class RateLimited(ProviderError):
    """The provider rejected the request because of rate limiting."""


class Timeout(ProviderError):
    """The provider call exceeded its deadline."""


class AuthError(ProviderError):
    """Authentication or authorization with the provider failed."""
