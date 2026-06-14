from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def save_vectors(output_dir: str, vectors: np.ndarray, mapping: pd.DataFrame) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)
    mapping.to_csv(out / "mapping.csv", index=False, encoding="utf-8-sig")
    print(f"Saved vectors: {out / 'vectors.npy'}")
    print(f"Saved mapping: {out / 'mapping.csv'}")
    print(f"Shape: {vectors.shape}")


def save_csv(path: str, df: pd.DataFrame, message: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{message}: {output_path}")
