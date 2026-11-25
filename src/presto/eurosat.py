"""
EuroSAT (multispectral) for Presto: GeoTIFF cache and on-device input construction.

``build_cache`` reads the 27,000 EuroSAT_MS GeoTIFFs once (rasterio + pyproj) into three
arrays (raw uint16 reflectances, per-image lon/lat vectors, labels); ``EuroSatCache`` reads
that cache, including caches built by the original case-study code. ``PrestoInputBuilder``
reproduces Presto's own EuroSAT pipeline: patch-mean the raw reflectances, scatter the 11
kept Sentinel-2 bands into Presto's 18-band array, normalize, drop B9, recompute NDVI.
Splits are the official EuroSAT splits (16,200 / 5,400 / 5,400).
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from . import bands as B

TIF_SUBDIR = Path("ds/images/remote_sensing/otherDatasets/sentinel_2/tif")
SPLITS = ("train", "val", "test")
PRESTO_WEIGHTS_URL = "https://raw.githubusercontent.com/nasaharvest/presto/main/data/default_model.pt"
SPLITS_URL = "https://hf.co/datasets/torchgeo/eurosat/resolve/main/eurosat-{}.txt"
EUROSAT_MS_URLS = ("https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip",
                   "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip")


def tif_root(root: Path) -> Path:
    """The folder holding the ten class folders. The Hugging Face archive nests it under
    ``TIF_SUBDIR``; other copies of EuroSAT_MS keep the class folders at the top or one level down."""
    root = Path(root)
    for cand in (root / TIF_SUBDIR, root / "EuroSAT_MS", root):
        if (cand / "AnnualCrop").is_dir():
            return cand
    hits = sorted(p.parent for p in root.rglob("AnnualCrop") if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"No EuroSAT_MS class folders (AnnualCrop/, Forest/, ...) under {root}")
    return hits[0]


def _tif_path(root: Path, name: str) -> Path:
    stem = name.strip()
    stem = stem[:-4] + ".tif" if stem.endswith(".jpg") else (stem if stem.endswith(".tif") else stem + ".tif")
    return root / stem.split("_")[0] / stem


def _read_one(args: Tuple[int, str]):
    import rasterio
    from pyproj import Transformer
    row, path = args
    with rasterio.open(path) as src:
        arr, tr, crs, h, w = src.read(), src.transform, src.crs, src.height, src.width
    xs = tr.c + tr.a * (np.arange(w) + 0.5) + tr.b * 0.5
    ys = tr.f + tr.d * 0.5 + tr.e * (np.arange(h) + 0.5)
    # Presto transforms the x and y vectors element-wise, then meshgrids the results.
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    return row, arr.astype(np.uint16), np.asarray(lon), np.asarray(lat)


def read_split_files(split_dir: Path) -> Dict[str, List[str]]:
    out = {}
    for split in SPLITS:
        path = Path(split_dir) / f"eurosat-{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file {path} (download from {SPLITS_URL.format(split)})")
        out[split] = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return out


def build_cache(eurosat_root: Path, split_dir: Path, cache_dir: Path, workers: int = 16) -> dict:
    """Reads every EuroSAT_MS GeoTIFF once and writes bands.npy, lonlat.npy, labels.npy."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = read_split_files(split_dir)
    names = [n for s in SPLITS for n in splits[s]]
    if len(set(names)) != len(names):
        raise ValueError("The official split files overlap")
    n = len(names)
    bands = np.lib.format.open_memmap(cache_dir / "bands.npy", mode="w+", dtype=np.uint16,
                                      shape=(n, len(B.S2_BANDS), B.IMAGE_SIZE, B.IMAGE_SIZE))
    lonlat = np.lib.format.open_memmap(cache_dir / "lonlat.npy", mode="w+", dtype=np.float32,
                                       shape=(n, 2, B.IMAGE_SIZE))
    labels = np.zeros(n, dtype=np.int64)
    tasks = []
    root = tif_root(eurosat_root)
    for row, name in enumerate(names):
        path = _tif_path(root, name)
        if not path.exists():
            raise FileNotFoundError(f"EuroSAT tif not found: {path}")
        tasks.append((row, str(path)))
        labels[row] = B.EUROSAT_LABELS[path.parent.name]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row, arr, lon, lat in pool.map(_read_one, tasks, chunksize=64):
            bands[row], lonlat[row, 0], lonlat[row, 1] = arr, lon, lat
    bands.flush()
    lonlat.flush()
    np.save(cache_dir / "labels.npy", labels)
    index, offset = {}, 0
    for split in SPLITS:
        index[split] = list(range(offset, offset + len(splits[split])))
        offset += len(splits[split])
    manifest = {"n_images": n, "image_size": B.IMAGE_SIZE, "n_tif_bands": len(B.S2_BANDS),
                "split_sizes": {s: len(splits[s]) for s in SPLITS}, "names": names,
                "split_index": index, "class_names": B.EUROSAT_CLASSES}
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def stratified_subset(indices: np.ndarray, labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Class-stratified subset, deterministic given the seed (e.g. 250 calibration images)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, size // len(classes))
    out = np.concatenate([rng.choice(indices[labels == c], size=min(per_class, int((labels == c).sum())),
                                     replace=False) for c in classes])
    if len(out) > size:
        out = rng.choice(out, size=size, replace=False)
    return np.sort(out)


class EuroSatCache:
    def __init__(self, cache_dir: Path, in_memory: bool = True):
        self.dir = Path(cache_dir)
        if not (self.dir / "manifest.json").exists():
            raise FileNotFoundError(f"No EuroSAT cache at {cache_dir}; build it with space_main.py build-cache")
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        mode = None if in_memory else "r"
        self.bands = np.load(self.dir / "bands.npy", mmap_mode=mode)
        self.lonlat = np.load(self.dir / "lonlat.npy", mmap_mode=mode)
        self.labels = np.load(self.dir / "labels.npy")
        self.split_index = {k: np.asarray(v, dtype=np.int64) for k, v in self.manifest["split_index"].items()}

    def indices(self, split: str, limit: Optional[int] = None, seed: int = 0) -> np.ndarray:
        idx = self.split_index[split]
        return idx if limit is None or limit >= len(idx) else stratified_subset(idx, self.labels[idx], limit, seed)


class PrestoInputBuilder:
    """Cached EuroSAT rows -> Presto encoder inputs, on the target device."""

    def __init__(self, cache: EuroSatCache, input_patch_size: int = 2, device: torch.device = torch.device("cpu")):
        if B.IMAGE_SIZE % input_patch_size:
            raise ValueError("input_patch_size must divide 64")
        self.cache, self.p, self.device = cache, input_patch_size, device
        self.n_per_dim = B.IMAGE_SIZE // input_patch_size
        self.n_patches = self.n_per_dim ** 2
        f = dict(dtype=torch.float32, device=device)
        self.add_by, self.divide_by = torch.tensor(B.ADD_BY, **f), torch.tensor(B.DIVIDE_BY, **f)
        self.keep = torch.tensor(B.KEEP_INDICES, device=device)
        self.kept_s2 = torch.tensor(B.KEPT_S2_TIF_IDX, device=device)
        self.kept_all = torch.tensor(B.KEPT_OVERALL_IDX, device=device)
        self.mask = torch.tensor(B.eurosat_channel_mask(), **f).view(1, 1, B.NUM_BANDS)

    def build(self, rows: np.ndarray) -> Dict[str, torch.Tensor]:
        rows = np.asarray(rows)
        raw = torch.from_numpy(np.ascontiguousarray(self.cache.bands[rows])).to(self.device)
        n, p, k = raw.shape[0], self.p, self.n_per_dim
        patched = raw.float().view(n, len(B.S2_BANDS), k, p, k, p).mean(dim=(3, 5))
        patched = patched.permute(0, 2, 3, 1).reshape(n, self.n_patches, len(B.S2_BANDS))
        org = torch.zeros(n, self.n_patches, B.NUM_ORG_BANDS, device=self.device)
        org[:, :, self.kept_all] = patched[:, :, self.kept_s2]
        x = ((org + self.add_by) / self.divide_by)[:, :, self.keep]
        num = x[:, :, B.B8_IDX] - x[:, :, B.B4_IDX]
        den = x[:, :, B.B8_IDX] + x[:, :, B.B4_IDX]
        x[:, :, B.NDVI_IDX] = torch.where(den > 0, num / den, torch.zeros_like(den))
        lonlat = torch.from_numpy(np.ascontiguousarray(self.cache.lonlat[rows])).to(self.device)
        lon = lonlat[:, 0].view(n, k, p).mean(-1)
        lat = lonlat[:, 1].view(n, k, p).mean(-1)
        latlons = torch.stack([lat.unsqueeze(2).expand(n, k, k).reshape(n, -1),
                               lon.unsqueeze(1).expand(n, k, k).reshape(n, -1)], dim=-1)
        return {"x": x.unsqueeze(2), "latlons": latlons,
                "dynamic_world": torch.full((n, self.n_patches, 1), B.DYNAMIC_WORLD_CLASS_AMOUNT,
                                            dtype=torch.long, device=self.device),
                "month": torch.full((n, self.n_patches), B.EUROSAT_START_MONTH, dtype=torch.long, device=self.device),
                "labels": torch.from_numpy(self.cache.labels[rows]).to(self.device)}

    def model_args(self, batch):
        mask = self.mask.expand(batch["x"].shape[0], 1, B.NUM_BANDS)
        return batch["x"], batch["dynamic_world"], batch["latlons"], mask, batch["month"]


def iterate_batches(builder: PrestoInputBuilder, rows: np.ndarray, images_per_batch: int, shuffle: bool = False,
                    generator: Optional[torch.Generator] = None) -> Iterator[Dict[str, torch.Tensor]]:
    order = np.arange(len(rows))
    if shuffle:
        order = order[torch.randperm(len(rows), generator=generator).numpy()]
    for start in range(0, len(order), images_per_batch):
        yield builder.build(rows[order[start:start + images_per_batch]])


def images_per_batch(n_patches: int, patch_budget: int = 40920) -> int:
    """Presto's batch convention: a fixed patch budget (39 images at 1,024 patches each)."""
    return max(1, patch_budget // n_patches)
