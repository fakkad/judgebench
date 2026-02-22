"""Bias detection modules for JudgeBench."""

from judgebench.bias.leniency import detect as detect_leniency
from judgebench.bias.position import detect as detect_position
from judgebench.bias.self_enhance import detect as detect_self_enhance
from judgebench.bias.verbosity import detect as detect_verbosity

__all__ = [
    "detect_position",
    "detect_verbosity",
    "detect_self_enhance",
    "detect_leniency",
]
