import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

def load_and_clean_data(file_path: str = None) -> pd.DataFrame:
    """
    Shared utility function to load, clean, and enrich the Nassau Candy dataset.
    This ensures both the CLI analysis and Streamlit dashboard use the exact same data rules.
    """
    # Auto-resolve the file path if not provided
    if not file_path:
        base_dir = Path(__file__).parent
        file_path = base_dir / "Nassau Candy Distributor.csv"
        if not file_path.exists():
            file_path = base_dir / "Nassau_Candy_Distributor.csv"
            if not file_path.exists():
                raise FileNotFoundError("Could not find the dataset CSV file in the directory.")

    df = pd.read_csv(file_path)

    # 1. Filter out invalid rows to avoid divide-by-zero and negative values
    df = df[df["Sales"] > 0]
    df = df[df["Gross Profit"] > 0]
    df = df[df["Units"] > 0]

    # 2. Normalize key string fields
    df["Product Name"] = df["Product Name"].str.strip()
    df["Division"] = df["Division"].str.strip()

    # 3. Convert dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    # 4. Derived metrics used across the project
    df["Gross_Margin_Pct"] = (df["Gross Profit"] / df["Sales"]) * 100
    df["Profit_Per_Unit"] = df["Gross Profit"] / df["Units"]
    
    # Shipping efficiency metrics
    df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    
    # Create both an object period and a string version
    df["Month_Period"] = df["Order Date"].dt.to_period("M")
    df["Month"] = df["Month_Period"].astype(str)
    df["Year"] = df["Order Date"].dt.year
    df["Quarter"] = df["Order Date"].dt.quarter

    # 5. Map products to their factory group (useful for factory-level analysis tabs)
    factory_map = {
        "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
        "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
        "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
        "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
        "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
        "Laffy Taffy": "Sugar Shack",
        "SweeTARTS": "Sugar Shack",
        "Nerds": "Sugar Shack",
        "Fun Dip": "Sugar Shack",
        "Fizzy Lifting Drinks": "Sugar Shack",
        "Everlasting Gobstopper": "Secret Factory",
        "Hair Toffee": "The Other Factory",
        "Lickable Wallpaper": "Secret Factory",
        "Wonka Gum": "Secret Factory",
        "Kazookles": "The Other Factory",
    }
    df["Factory"] = df["Product Name"].map(factory_map)
    # Default Group for missing mappings
    df["Factory"] = df["Factory"].fillna("General Logistics")

    return df
