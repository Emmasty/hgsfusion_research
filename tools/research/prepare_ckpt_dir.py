"""Point an HGSFusion run's ckpt directory at the mounted weight store."""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STORE = Path("/mnt/data2/zhangyan/hgsfusion_weights/runs")


def prepare(dataset, tag):
    local = ROOT / "output/cfgs/hgsfusion" / dataset / tag / "ckpt"
    target = STORE / dataset / tag
    if local.is_symlink():
        if local.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected link: {local}")
        print(f"already linked: {local} -> {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        entries = list(local.iterdir())
        for entry in entries:
            if not entry.is_symlink() or entry.resolve() != (target / entry.name).resolve():
                raise RuntimeError(f"Local ckpt directory has an unverified file: {entry}")
        backup = local.with_name("ckpt.migration-backup")
        if backup.exists():
            raise RuntimeError(f"Unresolved backup: {backup}")
        os.replace(local, backup)
    else:
        backup = None
    try:
        local.symlink_to(target, target_is_directory=True)
        if local.resolve() != target.resolve():
            raise RuntimeError(f"Directory link verification failed: {local}")
    except Exception:
        local.unlink(missing_ok=True)
        if backup is not None:
            os.replace(backup, local)
        raise
    if backup is not None:
        for entry in backup.iterdir():
            entry.unlink()
        backup.rmdir()
    print(f"linked: {local} -> {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-existing", action="store_true")
    parser.add_argument("--dataset", choices=("hgsfusion_vod", "hgsfusion_tj4d"))
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.all_existing:
        paths = sorted((ROOT / "output/cfgs/hgsfusion").glob("*/*/ckpt"))
        for path in paths:
            prepare(path.parent.parent.name, path.parent.name)
    elif args.dataset and args.tag:
        prepare(args.dataset, args.tag)
    else:
        parser.error("use --all-existing or both --dataset and --tag")


if __name__ == "__main__":
    main()
