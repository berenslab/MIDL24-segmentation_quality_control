"""Compare legacy and bunch-free checkpoints (maintainer utility).

Validates that converted checkpoints match the original weights for inference.
"""

from argparse import ArgumentParser
import subprocess
from pathlib import Path

import torch

from utils.checkpoint.io import load_checkpoint
from utils.model_definition import FR_UNet


def short_path(path: Path | str, tail: int = 2) -> str:
    """Show at most the last ``tail`` path components, hiding deep parents."""
    parts = Path(path).parts
    if len(parts) <= tail:
        return str(Path(*parts))
    return ".../" + "/".join(parts[-tail:])


def ensure_legacy_weights(
    old_dir: Path,
    repo_root: Path,
    pattern: str,
    new_weights_dir: Path,
) -> None:
    """Populate legacy checkpoints from git main if the folder is missing."""
    old_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(old_dir.glob(pattern))
    if existing:
        print(f"Using {len(existing)} legacy checkpoint(s) in {short_path(old_dir)}")
        return

    print(f"{short_path(old_dir)} empty — extracting legacy weights from git main...")
    for name in [p.name for p in sorted(new_weights_dir.glob(pattern))]:
        out = old_dir / name
        blob = subprocess.run(
            ["git", "show", f"main:trained/{name}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        out.write_bytes(blob)
        print(f"  wrote {short_path(out)} ({len(blob) / 1e6:.1f} MB)")


def load_legacy_checkpoint(path: Path, device: str = "cpu") -> dict:
    """Load a legacy checkpoint (requires bunch for unpickling)."""
    try:
        import bunch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Legacy checkpoints require `bunch` (`pip install bunch`)."
        ) from exc
    return torch.load(path, map_location=device, weights_only=False)


def build_model(device: str = "cpu") -> FR_UNet:
    model = FR_UNet(num_classes=1, num_channels=3, feature_scale=2, dropout=0.1)
    model.to(device)
    model.eval()
    return model


def state_dict_max_diff(a: dict, b: dict) -> float:
    if set(a.keys()) != set(b.keys()):
        missing_a = set(b.keys()) - set(a.keys())
        missing_b = set(a.keys()) - set(b.keys())
        raise KeyError(
            f"state_dict key mismatch: only in a={missing_b}, only in b={missing_a}"
        )
    diffs = [(a[k] - b[k]).abs().max().item() for k in a.keys()]
    return max(diffs)


def forward_max_diff(model_a: FR_UNet, model_b: FR_UNet, device: str = "cpu") -> float:
    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        y_a = model_a(x)
        y_b = model_b(x)
    return (y_a - y_b).abs().max().item()


def compare_checkpoints(
    old_weights_dir: Path,
    new_weights_dir: Path,
    pattern: str = "FRUNet_*.pth",
    device: str = "cpu",
) -> list[dict]:
    """Compare legacy and converted checkpoints, returning per-file results."""
    old_paths = sorted(old_weights_dir.glob(pattern))
    new_paths = sorted(new_weights_dir.glob(pattern))
    new_by_name = {p.name: p for p in new_paths}

    if not old_paths:
        raise FileNotFoundError(
            f"No legacy weights matching {pattern} in {short_path(old_weights_dir)}"
        )
    if not new_paths:
        raise FileNotFoundError(
            f"No new weights matching {pattern} in {short_path(new_weights_dir)}"
        )

    rows = []
    for old_path in old_paths:
        name = old_path.name
        if name not in new_by_name:
            rows.append({"file": name, "status": "MISSING in new dir"})
            continue

        new_path = new_by_name[name]
        old_mb = old_path.stat().st_size / 1e6
        new_mb = new_path.stat().st_size / 1e6

        old_ckpt = load_legacy_checkpoint(old_path, device=device)
        new_ckpt = load_checkpoint(new_path, device=device)

        keys_match = list(old_ckpt.keys()) == list(new_ckpt.keys())
        sd_diff = state_dict_max_diff(old_ckpt["state_dict"], new_ckpt["state_dict"])

        model_old = build_model(device)
        model_old.load_state_dict(old_ckpt["state_dict"])
        model_new = build_model(device)
        model_new.load_state_dict(new_ckpt["state_dict"])
        fwd_diff = forward_max_diff(model_old, model_new, device=device)

        ok = (
            keys_match
            and sd_diff == 0.0
            and fwd_diff == 0.0
            and isinstance(new_ckpt["config"], dict)
        )
        rows.append(
            {
                "file": name,
                "old_mb": round(old_mb, 1),
                "new_mb": round(new_mb, 1),
                "old_keys": list(old_ckpt.keys()),
                "new_keys": list(new_ckpt.keys()),
                "keys_match": keys_match,
                "config_type_old": type(old_ckpt["config"]).__name__,
                "config_type_new": type(new_ckpt["config"]).__name__,
                "state_dict_max_diff": sd_diff,
                "forward_max_diff": fwd_diff,
                "status": "OK" if ok else "MISMATCH",
            }
        )
    return rows


def print_report(rows: list[dict], old_weights_dir: Path, new_weights_dir: Path) -> int:
    """Print comparison report. Returns number of mismatches."""
    print("=" * 72)
    print("CHECKPOINT COMPARISON REPORT")
    print("=" * 72)
    print(f"Legacy: {short_path(old_weights_dir)}")
    print(f"New:    {short_path(new_weights_dir)}")
    print()

    for row in rows:
        print(f"--- {row['file']} ---")
        if row.get("status") == "MISSING in new dir":
            print("  STATUS: MISSING in new dir")
            continue
        print(f"  size:         {row['old_mb']} MB -> {row['new_mb']} MB")
        print(f"  keys:         {row['old_keys']}")
        print(f"  keys match:   {row['keys_match']}")
        print(f"  config type:  {row['config_type_old']} -> {row['config_type_new']}")
        print(f"  state_dict:   max|diff| = {row['state_dict_max_diff']:.2e}")
        print(f"  forward:      max|diff| = {row['forward_max_diff']:.2e}")
        print(f"  STATUS:       {row['status']}")
        print()

    ok = [r for r in rows if r.get("status") == "OK"]
    bad = [r for r in rows if r.get("status") not in ("OK",)]
    print("=" * 72)
    print(f"SUMMARY: {len(ok)}/{len(rows)} matched (identical keys, plain config, zero diffs)")
    if bad:
        print(f"Issues: {[r['file'] for r in bad]}")
    else:
        print("All pairs OK — new checkpoints preserve legacy keys without bunch.")
    print("=" * 72)
    return len(bad)


def main(args):
    repo_root = Path(args.repo_root).resolve()
    old_weights_dir = Path(args.old_weights_dir)
    if not old_weights_dir.is_absolute():
        old_weights_dir = repo_root / old_weights_dir
    new_weights_dir = Path(args.new_weights_dir)
    if not new_weights_dir.is_absolute():
        new_weights_dir = repo_root / new_weights_dir

    if args.ensure_legacy:
        ensure_legacy_weights(
            old_weights_dir,
            repo_root,
            args.pattern,
            new_weights_dir,
        )

    rows = compare_checkpoints(
        old_weights_dir,
        new_weights_dir,
        pattern=args.pattern,
        device=args.device,
    )
    n_bad = print_report(rows, old_weights_dir, new_weights_dir)
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root for resolving relative weight directories.",
    )
    parser.add_argument(
        "--old-weights-dir",
        type=str,
        default="trained_legacy",
        help="Directory with legacy bunch checkpoints.",
    )
    parser.add_argument(
        "--new-weights-dir",
        type=str,
        default="trained",
        help="Directory with converted checkpoints.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="FRUNet_*.pth",
        help="Glob pattern for checkpoint files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for forward-pass comparison.",
    )
    parser.add_argument(
        "--ensure-legacy",
        action="store_true",
        help="Extract legacy weights from git main if old dir is empty.",
    )
    main(parser.parse_args())
