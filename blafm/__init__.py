"""Bayesian latent functional mediation."""

from .effects import summarize_effects
from .model import fit

__all__ = ["fit", "summarize_effects"]
