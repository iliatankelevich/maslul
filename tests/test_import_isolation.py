"""Locks in the optional-extras guarantee: ``import maslul`` must not pull in any provider SDK,
so an install with only some extras (or none) still imports cleanly. Runs in a subprocess
because the rest of the test session imports every SDK, which would pollute ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_maslul_does_not_load_provider_sdks() -> None:
    code = (
        "import sys, maslul\n"
        "leaked = [m for m in ('anthropic', 'google.genai', 'xai_sdk') if m in sys.modules]\n"
        "assert not leaked, f'import maslul pulled in provider SDKs: {leaked}'\n"
        "assert hasattr(maslul, 'Router')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
