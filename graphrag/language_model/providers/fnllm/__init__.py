"""FNLLM provider module with runtime patches applied."""

from .patches import apply_patches

apply_patches()

__all__ = ["apply_patches"]
