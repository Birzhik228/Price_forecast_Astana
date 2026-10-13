"""Общие пути и настройки проекта."""

from pathlib import Path
from typing import Final


# Пути не зависят от папки, из которой запущен скрипт.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
RAW_LISTINGS_PATH: Final[Path] = RAW_DATA_DIR / "listings.csv"
KAGGLE_LISTINGS_PATH: Final[Path] = RAW_DATA_DIR / "kaggle_astana" / "astana_apartments.csv"
CLEAN_LISTINGS_PATH: Final[Path] = PROCESSED_DATA_DIR / "listings_clean.csv"
REJECTED_LISTINGS_PATH: Final[Path] = PROCESSED_DATA_DIR / "listings_rejected.csv"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
CLEANING_REPORT_PATH: Final[Path] = REPORTS_DIR / "cleaning_report.json"

# Фиксированная дата проекта; не меняется сама при повторном запуске.
REFERENCE_YEAR: Final[int] = 2026

# Общие параметры разбиения и обучения.
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
CV_FOLDS: Final[int] = 3
SEARCH_ITERATIONS: Final[int] = 8
MODEL_PIPELINE_PATH: Final[Path] = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Final[Path] = MODELS_DIR / "model_metadata.json"
PROPERTY_CATALOG_DIR: Final[Path] = PROCESSED_DATA_DIR / "property_catalog"
