"""Presto band definitions and EuroSAT -> Presto input construction constants.

Every constant here is copied verbatim from the Presto repository
(``presto/dataops/pipelines/s1_s2_era5_srtm.py``) so that our EuroSAT input
tensors are bit-for-bit identical to the ones produced by Presto's own
``presto/eval/eurosat_eval.py``.  See README.md ("Band mapping") for the
derivation and a statement of what each masked band contributes (nothing --
masked tokens are dropped inside the encoder).

Do not "clean up" the odd-looking values below (e.g. the S1 shift of 25.0 that
turns a zero-filled radar band into a constant 1.0): they are what Presto does,
and reproducing them exactly is what makes our FP32 baseline comparable to the
accuracies published in Presto's Table 5 / Table 11.
"""

from collections import OrderedDict
from typing import Dict, List

# --------------------------------------------------------------------------
# Raw band definitions (verbatim from Presto)
# --------------------------------------------------------------------------
S1_BANDS = ["VV", "VH"]
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]

# Sentinel-2 L1C band order, which is also the band order of the EuroSAT_MS
# GeoTIFFs (13 bands, uint16).
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S2_SHIFT_VALUES = [0.0] * len(S2_BANDS)
S2_DIV_VALUES = [1e4] * len(S2_BANDS)

ERA5_BANDS = ["temperature_2m", "total_precipitation"]
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]

SRTM_BANDS = ["elevation", "slope"]
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]

DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
STATIC_BANDS = SRTM_BANDS
DYNAMIC_BANDS_SHIFT = S1_SHIFT_VALUES + S2_SHIFT_VALUES + ERA5_SHIFT_VALUES
DYNAMIC_BANDS_DIV = S1_DIV_VALUES + S2_DIV_VALUES + ERA5_DIV_VALUES
STATIC_BANDS_SHIFT = SRTM_SHIFT_VALUES
STATIC_BANDS_DIV = SRTM_DIV_VALUES

# Presto's "Engineer" drops the two 60 m atmospheric bands.
REMOVED_BANDS = ["B1", "B10"]

# The 18-band "original" array that Presto's EuroSAT loader fills in.
BANDS = [x for x in DYNAMIC_BANDS if x not in REMOVED_BANDS] + STATIC_BANDS + ["NDVI"]
ADD_BY = (
    [DYNAMIC_BANDS_SHIFT[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + STATIC_BANDS_SHIFT
    + [0.0]
)
DIVIDE_BY = (
    [DYNAMIC_BANDS_DIV[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + STATIC_BANDS_DIV
    + [1.0]
)
NUM_ORG_BANDS = len(BANDS)  # 18

# B9 is dropped again at normalization time, leaving the 17 bands the encoder sees.
NORMED_BANDS = [x for x in BANDS if x != "B9"]
NUM_BANDS = len(NORMED_BANDS)  # 17

BANDS_GROUPS_IDX: "OrderedDict[str, List[int]]" = OrderedDict(
    {
        "S1": [NORMED_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [NORMED_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [NORMED_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [NORMED_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [NORMED_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [NORMED_BANDS.index(b) for b in ["B11", "B12"]],
        "ERA5": [NORMED_BANDS.index(b) for b in ERA5_BANDS],
        "SRTM": [NORMED_BANDS.index(b) for b in SRTM_BANDS],
        "NDVI": [NORMED_BANDS.index("NDVI")],
    }
)

# Index of B9 within the 18-band array (the band that normalize() drops).
KEEP_INDICES = [i for i, b in enumerate(BANDS) if b != "B9"]

NDVI_IDX = NORMED_BANDS.index("NDVI")
B8_IDX = NORMED_BANDS.index("B8")
B4_IDX = NORMED_BANDS.index("B4")

# --------------------------------------------------------------------------
# EuroSAT_MS GeoTIFF -> 18-band "original" array
# --------------------------------------------------------------------------
# Reproduces Presto's EuroSatDataset.image_to_eo_array: the 11 kept Sentinel-2
# bands are scattered into their BANDS positions and *everything else is left
# at zero* (there is no S1, ERA5 or SRTM data in EuroSAT).
_INDICES_TO_REMOVE = [S2_BANDS.index(b) for b in REMOVED_BANDS]
KEPT_S2_TIF_IDX = [i for i in range(len(S2_BANDS)) if i not in _INDICES_TO_REMOVE]
KEPT_OVERALL_IDX = [i for i, x in enumerate(BANDS) if (x in S2_BANDS and x not in REMOVED_BANDS)]

DYNAMIC_WORLD_CLASS_AMOUNT = 9  # value that marks Dynamic World as "missing"

EUROSAT_LABELS: Dict[str, int] = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}
EUROSAT_CLASSES = [c for c, _ in sorted(EUROSAT_LABELS.items(), key=lambda kv: kv[1])]

# Presto's EuroSAT loader hard-codes month 1 ("this is not the true start month!").
EUROSAT_START_MONTH = 1
IMAGE_SIZE = 64


def eurosat_channel_mask() -> List[int]:
    """Return the per-band mask Presto uses for EuroSAT (1 == masked).

    Reproduces ``EuroSatEval.update_mask`` for the multispectral case: every
    band group whose name contains "S2", plus NDVI, is visible; S1, ERA5 and
    SRTM are masked out and their tokens are dropped inside the encoder.
    """
    channels: List[int] = []
    for key, idx in BANDS_GROUPS_IDX.items():
        if "S2" in key:
            channels.extend(idx)
    channels.extend(BANDS_GROUPS_IDX["NDVI"])
    mask = [1] * NUM_BANDS
    for c in channels:
        mask[c] = 0
    return mask
