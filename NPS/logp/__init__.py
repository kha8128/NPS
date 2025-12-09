"""
NPS LogP: Log-Probability Foundation Model for Crystal Structure Analysis

A unified framework for:
- Crystal structure denoising
- Phase classification  
- Order parameter estimation

Reference:
    Kwon et al., "A log-probability foundation model for crystal structure 
    denoising, phase classification, and order parameters"
"""

__version__ = "0.1.0"

from .inference.denoise import denoise_structure, denoise_trajectory
from .inference.classify import classify_structure, classify_zeroshot
from .constants import STRUCTURE_TYPES, get_structure_types, get_common_name

__all__ = [
    "denoise_structure",
    "denoise_trajectory", 
    "classify_structure",
    "classify_zeroshot",
]
