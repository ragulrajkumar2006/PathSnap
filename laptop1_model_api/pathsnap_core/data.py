"""Loading the processed datasets for modelling."""
import pandas as pd

from .config import DATA_PROCESSED, TRAIN_STRIDE
from .features import FEATURES


def load_candidates(partition):
    return pd.read_parquet(DATA_PROCESSED / f"candidates_{partition}.parquet")


def load_fixes(partition):
    return pd.read_parquet(DATA_PROCESSED / f"fixes_{partition}.parquet")


def training_frame(stride=TRAIN_STRIDE):
    """Training rows: valid labels only, every `stride`-th fix of each trip."""
    df = load_candidates("train")
    df = df[(df.label_valid == 1) & (df.obs % stride == 0)].reset_index(drop=True)
    return df, df[FEATURES].to_numpy(float), df["label"].to_numpy(int), df["trip_id"].to_numpy()


def eval_frame(partition="student_test"):
    df = load_candidates(partition)
    fixes = load_fixes(partition)
    return df, fixes
