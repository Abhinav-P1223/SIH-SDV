"""Generate the MATLAB / Simulink export files (UNTESTED IN MATLAB; see matlab/README.md).

    python scripts/export_matlab.py                # -> matlab/
    python scripts/export_matlab.py --out build/m  # elsewhere
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matlab_export import export  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="matlab", help="output directory (default: matlab/)")
    args = ap.parse_args()
    for path in export(args.out):
        print("wrote", path)
    print("NOTE: generated files are untested in MATLAB (no installation available).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
