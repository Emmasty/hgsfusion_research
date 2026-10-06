"""Read-only audit of the two mounted datasets; needs only Python's stdlib."""

import argparse
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "vod": (ROOT / "data/vod_radar_5frames", 7),
    "vod_lidar": (ROOT / "data/vod_lidar", 4),
    "tj4d": (ROOT / "data/tj4d", 8),
}
VIRTUAL_DIR = "mask_maskformer_with_label_k_1_gauss_k_4_uniform"


def inspect(name, root, channels):
    result = {"root": str(root.resolve()), "exists": root.is_dir(), "channels": channels}
    if not root.is_dir():
        return result
    splits = {}
    for split in ("train", "val", "test"):
        path = root / "ImageSets" / (split + ".txt")
        if path.is_file():
            lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            splits[split] = {"count": len(lines), "unique": len(set(lines)), "first": lines[0] if lines else None}
    result["splits"] = splits
    directories = {}
    for section in ("training", "testing"):
        base = root / section
        if not base.is_dir():
            continue
        subdirs = {}
        for subdir in ("velodyne", "velodyne3", "image_2", "calib", "label_2", VIRTUAL_DIR):
            path = base / subdir
            if path.is_dir():
                files = sorted(p for p in path.iterdir() if p.is_file())
                subdirs[subdir] = {"count": len(files), "first": files[0].name if files else None}
        directories[section] = subdirs
    result["directories"] = directories
    for split in ("train", "val"):
        path = root / "ImageSets" / (split + ".txt")
        if not path.is_file():
            continue
        sample_ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
        folder = root / "training"
        missing = {}
        for subdir, extension in (("velodyne", ".bin"), ("calib", ".txt"), ("image_2", ".jpg" if name.startswith("vod") else ".png"), (VIRTUAL_DIR, ".pkl.npy")):
            if name == "vod_lidar" and subdir == VIRTUAL_DIR:
                continue
            absent = [sample for sample in sample_ids if not (folder / subdir / (sample + extension)).is_file()]
            missing[subdir] = {"count": len(absent), "examples": sorted(absent)[:5]}
        result[split + "_missing"] = missing
    point_dir = root / "training/velodyne"
    if point_dir.is_dir():
        point_files = sorted(point_dir.glob("*.bin"))
        if point_files:
            path = point_files[0]
            size = path.stat().st_size
            record_size = 4 * channels
            with path.open("rb") as f:
                record = f.read(record_size)
            result["point_sample"] = {
                "file": path.name, "bytes": size, "record_aligned": size % record_size == 0,
                "points": size // record_size,
                "first_point": list(struct.unpack("<" + "f" * channels, record)) if len(record) == record_size else None,
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    report = {name: inspect(name, *settings) for name, settings in DATASETS.items()}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
