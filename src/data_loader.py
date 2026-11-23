"""Data loading and schema definitions for UCI Forest Covertype dataset."""

from pathlib import Path
from typing import List, Optional
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area_1",
    "Wilderness_Area_2",
    "Wilderness_Area_3",
    "Wilderness_Area_4",
] + [f"Soil_Type_{i}" for i in range(1, 41)]

TARGET_COLUMN: str = "Cover_Type"
ALL_COLUMNS: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]

COVER_TYPE_LABELS = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Load raw Covertype CSV and assign standard column headers.

    Args:
        csv_path: Path to raw dataset CSV file.

    Returns:
        pd.DataFrame with named columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(path, header=None)
    if df.shape[1] == len(ALL_COLUMNS):
        df.columns = ALL_COLUMNS
    return df
