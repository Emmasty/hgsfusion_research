"""Copy research weights to a mounted disk, verify SHA256, then link old paths.

Run from any working directory. The default mode only prints the inventory.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}
DEFAULT_DEST = Path("/mnt/data2/zhangyan/hgsfusion_weights")


def weight_paths():
    paths = [ROOT / "hgsfusion_vod.pth", ROOT / "hgsfusion_tj4d.pth"]
    paths.extend((ROOT / "checkpoints").glob("*"))
    paths.extend((ROOT.parent / "checkpoints").glob("*"))
    paths.extend((ROOT / "output").rglob("*"))
    # Path.rglob does not descend into ckpt directories once they are symlinks.
    for ckpt_dir in (ROOT / "output/cfgs/hgsfusion").glob("*/*/ckpt"):
        paths.extend(ckpt_dir.glob("*"))
    return sorted({p for p in paths if p.suffix.lower() in SUFFIXES and (p.is_file() or p.is_symlink())})


def destination(source, dest_root):
    if source.parent in {ROOT / "checkpoints", ROOT.parent / "checkpoints"}:
        return dest_root / "pretrained" / source.name
    if source.parent == ROOT:
        return dest_root / "official" / source.name
    relative = source.relative_to(ROOT / "output")
    parts = relative.parts
    if len(parts) < 5 or parts[-2] != "ckpt":
        raise ValueError(f"Unexpected checkpoint layout: {source}")
    return dest_root / "runs" / parts[-4] / parts[-3] / parts[-1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source, target):
    source_hash = sha256(source)
    if target.exists():
        if target.stat().st_size != source.stat().st_size or sha256(target) != source_hash:
            raise RuntimeError(f"Existing destination differs: {target}")
        return source_hash
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    shutil.copy2(source, temporary)
    if sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Hash mismatch after copy: {source}")
    os.replace(temporary, target)
    return source_hash


def link_verified(source, target, expected_hash):
    if source.resolve() == target.resolve():
        return
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected existing symlink: {source}")
        return
    backup = source.with_name(source.name + ".migration-backup")
    if backup.exists():
        raise RuntimeError(f"Unresolved previous backup: {backup}")
    if sha256(target) != expected_hash:
        raise RuntimeError(f"Destination changed before linking: {target}")
    os.replace(source, backup)
    try:
        source.symlink_to(target)
        if source.resolve() != target.resolve() or sha256(source) != expected_hash:
            raise RuntimeError(f"Link verification failed: {source}")
    except Exception:
        source.unlink(missing_ok=True)
        os.replace(backup, source)
        raise
    backup.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--mode", choices=("plan", "copy", "link"), default="plan")
    args = parser.parse_args()
    dest_root = args.dest.resolve()
    sources = weight_paths()
    entries = []
    print(f"mode={args.mode} files={len(sources)} dest={dest_root}", flush=True)
    for number, source in enumerate(sources, 1):
        target = destination(source, dest_root)
        size = source.stat().st_size
        entry = {"source": str(source), "destination": str(target), "bytes": size}
        if args.mode != "plan":
            entry["sha256"] = copy_verified(source, target)
            if args.mode == "link":
                link_verified(source, target, entry["sha256"])
        entries.append(entry)
        print(f"{number}/{len(sources)} {source} -> {target}", flush=True)
    if args.mode != "plan":
        manifest = dest_root / "manifest.json"
        temporary = manifest.with_name("manifest.json.part")
        temporary.write_text(json.dumps(entries, indent=2) + "\n")
        os.replace(temporary, manifest)
        print(f"verified manifest: {manifest}", flush=True)
    else:
        print(f"total_bytes={sum(entry['bytes'] for entry in entries)}", flush=True)


if __name__ == "__main__":
    main()
