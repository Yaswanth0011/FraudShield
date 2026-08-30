"""
SentinelShield AI - Utility & Formatting Functions
"""

import random
import pandas as pd
from typing import Optional, List
from src.config import DEFAULT_MERCHANTS, DEFAULT_CITIES, DEFAULT_CARD_NETWORKS


def generate_synthetic_card_number(network: str = "Visa") -> str:
    """Generate a realistic masked or unmasked 16-digit card number."""
    if network == "American Express":
        digits = "37" + "".join(random.choices("0123456789", k=13))
    elif network == "Mastercard":
        digits = "5" + "".join(random.choices("12345", k=1)) + "".join(random.choices("0123456789", k=14))
    else:
        digits = "4" + "".join(random.choices("0123456789", k=15))
    return digits


def mask_card_number(card_number: str) -> str:
    """Format card number into masked format: **** **** **** 1234."""
    clean = "".join(filter(str.isdigit, str(card_number)))
    if len(clean) >= 4:
        last4 = clean[-4:]
        return f"•••• •••• •••• {last4}"
    return "•••• •••• •••• 9999"


def format_currency(amount: float, symbol: str = "$") -> str:
    """Format transaction amount into standardized currency display."""
    return f"{symbol}{amount:,.2f}"


def format_time_delta(seconds: float) -> str:
    """Convert elapsed transaction seconds to human-readable format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"+{hours:02d}h {minutes:02d}m {secs:02d}s"


def enrich_with_synthetic_metadata(df: pd.DataFrame, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Enrich raw transaction features with realistic synthetic merchant & customer metadata.
    """
    if seed is not None:
        random.seed(seed)

    try:
        from faker import Faker
        fake = Faker()
        Faker.seed(seed)
        names = [fake.name() for _ in range(len(df))]
        cities = [f"{fake.city()}, {fake.country_code()}" for _ in range(len(df))]
        merchants = [f"{fake.company()} {random.choice(['Retail', 'Online', 'Direct', 'Services'])}" for _ in range(len(df))]
    except ImportError:
        # Fallback names and locations if faker is unavailable
        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Casey", "Jesse", "Daniel", "Elena", "Marcus", "Sophia", "Liam", "Olivia", "Noah", "Emma"]
        last_names = ["Chen", "Smith", "Johnson", "Williams", "Brown", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Kumar", "Patel", "Müller", "Schneider", "Takahashi"]
        names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(len(df))]
        cities = [random.choice(DEFAULT_CITIES) for _ in range(len(df))]
        merchants = [random.choice(DEFAULT_MERCHANTS) for _ in range(len(df))]

    card_networks = [random.choice(DEFAULT_CARD_NETWORKS) for _ in range(len(df))]
    card_numbers = [generate_synthetic_card_number(net) for net in card_networks]
    transaction_ids = [f"TXN-{10001 + i}" for i in range(len(df))]

    enriched = df.copy()
    enriched["Transaction_ID"] = transaction_ids
    enriched["Customer_Name"] = names
    enriched["Card_Network"] = card_networks
    enriched["Card_Number"] = card_numbers
    enriched["Masked_Card"] = [mask_card_number(c) for c in card_numbers]
    enriched["Merchant"] = merchants
    enriched["Location"] = cities

    return enriched
