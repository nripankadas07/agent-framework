"""Measure local planner/tool overhead for the demo agent.

This prints numbers for the current machine only. Do not paste them into the
README as universal benchmark claims.
"""

from __future__ import annotations

import io
import statistics
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.no_api_key_agent import main


def run() -> None:
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            main()
        samples.append((time.perf_counter() - start) * 1000)
    mean_ms = statistics.mean(samples)
    print(
        f"runs={len(samples)} "
        f"mean_ms={mean_ms:.3f} "
        f"min_ms={min(samples):.3f} "
        f"max_ms={max(samples):.3f}"
    )


if __name__ == "__main__":
    run()
