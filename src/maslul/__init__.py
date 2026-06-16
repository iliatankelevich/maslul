"""maslul — an async, fully-typed LLM router across Anthropic, Gemini, and Grok.

Wraps multiple providers behind one call and routes each request to a model *tier* by
difficulty; the caller can always pin a level or an exact ``provider:model``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from maslul.config import RouterConfig
from maslul.errors import (
    AuthError,
    ConfigError,
    MaslulError,
    ProviderError,
    RateLimited,
    Timeout,
)
from maslul.providers.base import Provider
from maslul.router import ESCALATE_SENTINEL, Router, default_hard_signal
from maslul.types import (
    KNOWN_PROVIDERS,
    BypassPredicate,
    Classifier,
    CompleteHook,
    ErrorHook,
    HardSignal,
    JsonSchema,
    Level,
    MediaPart,
    Message,
    ModelSpec,
    ModelUsage,
    Request,
    Response,
    Role,
    RouteHook,
    RoutingDecision,
    Strategy,
    ToolCall,
    ToolDef,
    ToolExecutor,
    Usage,
)

try:
    __version__ = version("maslul")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "ESCALATE_SENTINEL",
    "KNOWN_PROVIDERS",
    "AuthError",
    "BypassPredicate",
    "Classifier",
    "CompleteHook",
    "ConfigError",
    "ErrorHook",
    "HardSignal",
    "JsonSchema",
    "Level",
    "MaslulError",
    "MediaPart",
    "Message",
    "ModelSpec",
    "ModelUsage",
    "Provider",
    "ProviderError",
    "RateLimited",
    "Request",
    "Response",
    "Role",
    "RouteHook",
    "Router",
    "RouterConfig",
    "RoutingDecision",
    "Strategy",
    "Timeout",
    "ToolCall",
    "ToolDef",
    "ToolExecutor",
    "Usage",
    "default_hard_signal",
    "__version__",
]
