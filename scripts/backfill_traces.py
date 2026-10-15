#!/usr/bin/env python3
"""Reconstruct a conservative Langfuse-compatible spool from RSI run files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.trace_backfill import write_backfill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Archived result/ directory")
    parser.add_argument("--output", required=True, help="New spool directory (must not exist)")
    parser.add_argument("--scope", choices=("diagnostic", "all"), default="diagnostic",
                        help="Six historical diagnostic candidates, or all train/validation answer logs")
    args = parser.parse_args()
    index = write_backfill(args.run, args.output, scope=args.scope)
    # Never print trace case IDs, query text, answers, or patient IDs.
    print(json.dumps({"output": str(Path(args.output)),
                      "decision_trace_id": index["decision_trace_id"],
                      "trace_count": index["trace_count"],
                      "observation_count": index["observation_count"],
                      "case_trace_count": index["case_trace_count"],
                      "selected_diagnostic_case_count": index["selected_diagnostic_case_count"],
                      "scope": index["scope"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
