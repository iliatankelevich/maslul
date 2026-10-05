"""Deprecated: the ``grok`` provider was renamed ``xai`` in 0.6.0. Import
:class:`~maslul.providers.xai.XAIProvider` from :mod:`maslul.providers.xai` instead."""

from __future__ import annotations

import warnings

from maslul.providers.xai import XAIProvider

warnings.warn(
    "maslul.providers.grok is deprecated; use maslul.providers.xai.XAIProvider",
    DeprecationWarning,
    stacklevel=2,
)

GrokProvider = XAIProvider

__all__ = ["GrokProvider"]
