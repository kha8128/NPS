"""
Constants and structure type definitions for LogP models.
"""

from pathlib import Path
from typing import List

# AFLOW prototype structure types (40 classes)
STRUCTURE_TYPES = [
    "A_cP240_205",
    "A_cI16_206",
    "A_hP3_191",
    "A_oF8_70",
    "A_cP1_221",
    "A_cP8_198",
    "A_oF128_70",
    "A_hP2_194",
    "A_oC24_63",
    "A_cP46_223",
    "A_cF240_202",
    "A_tI8_139",
    "A_hP3_152",
    "A_hR6_148",
    "A_hR3_166",
    "A_cI2_229",
    "A_cP8_205",
    "A_cF4_225",
    "A_tI4_141",
    "A_tP4_129",
    "A_cF8_227",
    "A_cI58_217",
    "A_tP4_136",
    "A_mP32_14",
    "A_tP30_136",
    "A_hR12_166",
    "A_hR2_166",
    "A_hR105_166",
    "A_mC16_12",
    "A_hR1_166",
    "A_aP24_2",
    "A_hP4_186",
    "A_tP50_134",
    "A_oC8_64",
    "A_tI2_139",
    "A_hP1_191",
    "A_cP20_213",
    "A_hP6_164",
    "A_oC4_63",
    "A_hP4_194",
]

# Common name mappings (AFLOW prototype -> common name)
COMMON_NAMES = {
    "A_cI2_229": "bcc",
    "A_cF4_225": "fcc", 
    "A_hP2_194": "hcp",
    "A_hP4_194": "dhcp",
    "A_cP1_221": "simple_cubic",
    "A_tI2_139": "bct",
    "A_hR1_166": "omega",
    "A_cF8_227": "diamond",
}

# Reverse mapping
AFLOW_FROM_COMMON = {v: k for k, v in COMMON_NAMES.items()}


def get_structure_types() -> List[str]:
    """Return the list of structure types."""
    return STRUCTURE_TYPES.copy()


def get_structure_index(name: str) -> int:
    """Get the index of a structure type (accepts common names like 'bcc')."""
    if name in AFLOW_FROM_COMMON:
        name = AFLOW_FROM_COMMON[name]
    return STRUCTURE_TYPES.index(name)


def get_common_name(aflow_name: str) -> str:
    """Get common name for an AFLOW prototype (if available)."""
    return COMMON_NAMES.get(aflow_name, aflow_name)
