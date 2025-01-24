# -*- coding: utf-8 -*-

"""exchange library for crypto trading"""

# -----------------------------------------------------------------------------

__version__ = "1.0.0"

# -----------------------------------------------------------------------------

from .bithumb import bithumb  # noqa: F401
from .coinone import coinone  # noqa: F401
from .gateio import gateio  # noqa: F401
from .kucoin import kucoin  # noqa: F401
from .upbit import upbit  # noqa: F401


exchanges = ["bithumb", "coinone", "gateio", "kucoin", "upbit"]

__all__ = exchanges
