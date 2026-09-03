"""Create the reproducible V2 train/test datasets."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_risk.config import FEATURES, RANDOM_STATE, TARGET
from credit_risk.data import create_train_test_split, load_modeling_data


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "df_sample.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Load raw data, reproduce the V2 split and persist its four parts."""
    data = load_modeling_data(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = create_train_test_split(
        data,
        test_size=0.20,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("Preprocessing completed")
    print(f"  Source: {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Features: {len(FEATURES)} | Target: {TARGET}")
    print(f"  Split random state: {RANDOM_STATE}")
    print(f"  Train: {X_train.shape} | default rate: {y_train.mean():.4%}")
    print(f"  Test:  {X_test.shape} | default rate: {y_test.mean():.4%}")
    print(f"  Output: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
