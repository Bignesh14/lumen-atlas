# Lumen Atlas - NASA Space Apps Challenge 2026 (Lunar Landing Site Navigator)
# AI-assisted code: written with Claude (Anthropic), reviewed and run by Bignesh Das.
# License: Apache-2.0
"""Paths and physical constants shared by every module."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Data lives next to the repo (never committed): SpaceApps/data.nosync/
DATA = Path(os.environ.get("LUMEN_DATA", REPO.parent / "data.nosync"))
DEM_DIR = DATA / "dem"
KERNEL_DIR = DATA / "kernels"
REF_DIR = DATA / "reference"
OUT_DIR = DATA / "out"          # big intermediate products (horizons, maps)
WEB_ASSETS = REPO / "web" / "assets"  # small products that ship with the web app

R_MOON_M = 1737400.0            # LOLA reference sphere radius (m), from the LBL files
MAST_M = 2.0                    # observer height above ground (solar panel / antenna), m
SUN_RADIUS_KM = 695700.0        # IAU nominal solar radius
