
"""PathSnap dataset quality and exploration utilities."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_trip(trip_dir):
    """Inspect one trip without changing its source data."""
    trip_dir = Path(trip_dir)
    result = {"trip": trip_dir.name}

    specs = {
        "track": ["longitude", "latitude", "timestamp"],
        "nodes": ["longitude", "latitude"],
        "arcs": ["from_node", "to_node"],
    }

    for extension, columns in specs.items():
        files = sorted(trip_dir.glob(f"*.{extension}"))
        result[f"{extension}_present"] = bool(files)

        if not files:
            result[f"{extension}_rows"] = 0
            continue

        try:
            df = pd.read_csv(
                files[0],
                sep=r"\s+",
                header=None,
                names=columns,
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            result[f"{extension}_error"] = str(exc)
            continue

        result[f"{extension}_rows"] = len(df)
        result[f"{extension}_missing_values"] = int(
            df.isna().sum().sum()
        )

        if extension == "track":
            lon = pd.to_numeric(df["longitude"], errors="coerce")
            lat = pd.to_numeric(df["latitude"], errors="coerce")
            times = pd.to_numeric(df["timestamp"], errors="coerce")

            invalid = (
                ~np.isfinite(lon)
                | ~np.isfinite(lat)
                | ~lon.between(-180, 180)
                | ~lat.between(-90, 90)
            )

            result["invalid_gps_points"] = int(invalid.sum())
            result["duplicate_timestamps"] = int(
                times.dropna().duplicated().sum()
            )

            deltas = times.diff()
            result["out_of_order_timestamps"] = int(
                (deltas < 0).sum()
            )
            result["gaps_over_5_seconds"] = int(
                (deltas > 5).sum()
            )
            result["max_gap_seconds"] = (
                float(deltas[deltas > 0].max())
                if (deltas > 0).any()
                else None
            )

    result["route_present"] = bool(
        list(trip_dir.glob("*.route"))
    )

    return result


def analyze_dataset(data_dir, output):
    """Analyze all trip directories and export a CSV summary."""
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    trip_dirs = sorted(
        path for path in data_dir.iterdir()
        if path.is_dir()
    )

    if not trip_dirs:
        raise ValueError("No trip directories found.")

    results = [analyze_trip(path) for path in trip_dirs]
    summary = pd.DataFrame(results)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(f"Trips analyzed: {len(summary)}")
    print(f"Summary saved to: {output}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore PathSnap GPS dataset quality"
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing trip folders",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset_quality.csv",
    )

    args = parser.parse_args()
    analyze_dataset(args.data_dir, args.output)
