"""
SentinelShield AI - Transaction Simulation & Synthetic Metadata Generator
"""

import argparse
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_PATH,
    SIMULATED_DATA_PATH,
    FEATURE_COLS,
    TARGET_COL
)
from src.utils import enrich_with_synthetic_metadata


def generate_simulation_dataset(
    source_path: str = str(RAW_DATA_PATH),
    output_path: str = str(SIMULATED_DATA_PATH),
    n_fraud: int = 492,
    n_legit: int = 1000,
    seed: int = 42
):
    """
    Sample a balanced/representative cohort of legitimate and fraudulent transactions
    and enrich with realistic synthetic metadata.
    """
    source = Path(source_path)
    output = Path(output_path)

    if not source.exists():
        if output.exists():
            print(f"Source file '{source}' not found. Enriching existing simulated file '{output}'...")
            df = pd.read_csv(output)
        else:
            raise FileNotFoundError(f"Neither source file '{source}' nor output file '{output}' exists.")
    else:
        print(f"Reading raw transactions from '{source}'...")
        df = pd.read_csv(source)

        if TARGET_COL in df.columns:
            fraud_subset = df[df[TARGET_COL] == 1]
            legit_subset = df[df[TARGET_COL] == 0]

            n_fraud_sample = min(len(fraud_subset), n_fraud)
            n_legit_sample = min(len(legit_subset), n_legit)

            print(f"Sampling {n_fraud_sample} fraud records and {n_legit_sample} legitimate records...")
            sampled_fraud = fraud_subset.sample(n=n_fraud_sample, random_state=seed)
            sampled_legit = legit_subset.sample(n=n_legit_sample, random_state=seed)

            df = pd.concat([sampled_fraud, sampled_legit]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print("Enriching transactions with synthetic identities, merchants, and card data...")
    enriched_df = enrich_with_synthetic_metadata(df, seed=seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output, index=False)
    print(f"✅ Successfully generated simulation dataset: {output} ({len(enriched_df)} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SentinelShield Simulated Transaction Dataset")
    parser.add_argument("--source", type=str, default=str(RAW_DATA_PATH), help="Source creditcard.csv path")
    parser.add_argument("--output", type=str, default=str(SIMULATED_DATA_PATH), help="Output CSV path")
    parser.add_argument("--fraud", type=int, default=492, help="Number of fraud transactions to sample")
    parser.add_argument("--legit", type=int, default=1000, help="Number of legitimate transactions to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    generate_simulation_dataset(
        source_path=args.source,
        output_path=args.output,
        n_fraud=args.fraud,
        n_legit=args.legit,
        seed=args.seed
    )
