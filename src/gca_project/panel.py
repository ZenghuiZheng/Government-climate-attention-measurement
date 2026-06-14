import pandas as pd

from gca_project.io_utils import save_csv
from gca_project.text_utils import ensure_columns


def merge_panel_data(panel_csv: str, similarity_csv: str, output_csv: str, key_columns: list[str]) -> None:
    panel = pd.read_csv(panel_csv)
    similarity = pd.read_csv(similarity_csv)
    ensure_columns(panel.columns, key_columns)
    ensure_columns(similarity.columns, key_columns)

    merged = panel.merge(similarity, on=key_columns, how="left")
    save_csv(output_csv, merged, "Saved merged panel data")
