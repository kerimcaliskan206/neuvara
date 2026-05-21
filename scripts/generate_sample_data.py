"""
Generate a synthetic hantavirus prediction dataset for development and testing.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --samples 2000 --output custom.csv

Output: data/raw/hantavirus_sample.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


def generate(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # ── Features ─────────────────────────────────────────────────────────────
    age = rng.integers(18, 75, size=n_samples).astype(float)
    gender = rng.choice(["M", "F"], size=n_samples)
    region = rng.choice(["north", "south", "east", "west", "central"], size=n_samples)
    season = rng.choice(["spring", "summer", "fall", "winter"], size=n_samples)

    rodent_contact = rng.integers(0, 2, size=n_samples)
    outdoor_work = rng.integers(0, 2, size=n_samples)
    fever = rng.integers(0, 2, size=n_samples)
    myalgia = rng.integers(0, 2, size=n_samples)      # muscle pain
    headache = rng.integers(0, 2, size=n_samples)
    thrombocytopenia = rng.integers(0, 2, size=n_samples)  # low platelets (HPS marker)

    rodent_density = rng.uniform(0.0, 10.0, size=n_samples).round(2)
    precipitation_mm = rng.uniform(0.0, 500.0, size=n_samples).round(1)
    humidity_pct = rng.uniform(30.0, 95.0, size=n_samples).round(1)

    # ── Label generation (epidemiologically motivated) ────────────────────────
    # Risk score: higher → more likely to be positive
    risk_score = (
        0.40 * rodent_contact
        + 0.25 * fever
        + 0.20 * myalgia
        + 0.15 * thrombocytopenia
        + 0.10 * outdoor_work
        + 0.03 * (rodent_density / 10.0)
        + 0.02 * (humidity_pct / 100.0)
        + 0.05 * np.isin(season, ["spring", "summer"]).astype(float)
        + 0.03 * np.isin(region, ["north", "west"]).astype(float)
    )
    # Add noise and convert to probability
    noise = rng.normal(0, 0.08, size=n_samples)
    probability = 1 / (1 + np.exp(-(risk_score + noise - 0.6) * 5))

    label = (probability > rng.uniform(size=n_samples)).astype(int)

    # ── Introduce realistic missingness (3–8%) ────────────────────────────────
    def add_missing(arr, rate: float):
        mask = rng.random(size=len(arr)) < rate
        result = arr.astype(object)
        result[mask] = np.nan
        return result

    df = pd.DataFrame({
        "age": add_missing(age, 0.03),
        "gender": add_missing(gender, 0.02),
        "region": region,
        "season": season,
        "rodent_contact": rodent_contact,
        "outdoor_work": outdoor_work,
        "fever": fever,
        "myalgia": myalgia,
        "headache": headache,
        "thrombocytopenia": thrombocytopenia,
        "rodent_density": add_missing(rodent_density, 0.05),
        "precipitation_mm": add_missing(precipitation_mm, 0.04),
        "humidity_pct": add_missing(humidity_pct, 0.03),
        "label": label,
    })

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic hantavirus prediction data."
    )
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--output", type=str, default="hantavirus_sample.csv", help="Output filename"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = _PROJECT_ROOT / "data" / "raw" / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples} samples (seed={args.seed})...")
    df = generate(n_samples=args.samples, random_state=args.seed)

    positive_rate = df["label"].mean() * 100
    print(f"  Positive rate : {positive_rate:.1f}%")
    print(f"  Missing values: {df.isnull().sum().sum()} cells")
    print(f"  Features      : {len(df.columns) - 1}")

    df.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
