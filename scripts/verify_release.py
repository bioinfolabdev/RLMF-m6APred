from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("H_b", "H_k", "H_l", "M_b", "M_h", "M_k", "M_l", "M_t", "R_b", "R_k", "R_l")


def main() -> None:
    failures: list[str] = []
    for dataset in DATASETS:
        for fold in range(1, 6):
            path = ROOT / "weights" / dataset / f"fold_{fold:02d}.pt"
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty fold checkpoint: {dataset}/fold_{fold:02d}.pt")
        for role in ("internal_benchmark", "independent_test"):
            path = ROOT / "data" / role / f"{dataset}.csv"
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty dataset: {role}/{dataset}.csv")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("Release files are present: 55 fold checkpoints and 22 dataset tables.")


if __name__ == "__main__":
    main()
