"""Convert legacy FR-UNet checkpoints to plain torch dicts without bunch (maintainer utility).

Legacy checkpoints pickle a nested ``bunch.Bunch`` in the ``config`` field.
Converted checkpoints keep the same top-level keys (``arch``, ``epoch``,
``state_dict``, ``optimizer``, ``config``) with all values plainized.

Requires ``bunch`` only for reading the legacy files (``pip install bunch``).
"""

from argparse import ArgumentParser
from pathlib import Path

import torch

from .io import convert_legacy_checkpoint


def convert_checkpoint(src: Path, dst: Path) -> dict:
    """Convert one legacy checkpoint to a bunch-free checkpoint."""
    try:
        import bunch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Loading legacy checkpoints requires `bunch` (`pip install bunch`)."
        ) from exc
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    converted = convert_legacy_checkpoint(ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, dst)
    return {
        "source_mb": src.stat().st_size / 1e6,
        "target_mb": dst.stat().st_size / 1e6,
        "source_keys": list(ckpt.keys()),
        "target_keys": list(converted.keys()),
    }


def main(args):
    """Convert all FRUNet checkpoints in a directory."""
    src_dir = Path(args.source_dir)
    dst_dir = Path(args.output_dir) if args.output_dir else src_dir
    pattern = args.pattern

    checkpoints = sorted(src_dir.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching {pattern} in {src_dir}")

    for src in checkpoints:
        dst = dst_dir / src.name
        if args.in_place:
            tmp = dst.with_suffix(".tmp.pth")
            info = convert_checkpoint(src, tmp)
            tmp.replace(dst)
        else:
            info = convert_checkpoint(src, dst)
        print(
            f"{src.name}: keys={info['source_keys']} -> {info['target_keys']} "
            f"({info['source_mb']:.1f}MB -> {info['target_mb']:.1f}MB)"
        )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=str,
        default="trained_legacy",
        help="Directory containing legacy .pth checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained",
        help="Output directory for converted checkpoints.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="FRUNet_*.pth",
        help="Glob pattern for checkpoint files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace checkpoints in --source-dir (ignores --output-dir).",
    )
    main(parser.parse_args())
