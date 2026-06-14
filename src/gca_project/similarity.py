import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from gca_project.io_utils import save_csv
from gca_project.text_utils import ensure_columns


def aggregate_similarity(
    sentence_vectors_path: str,
    sentence_mapping_path: str,
    phrase_vectors_path: str,
    output_csv: str,
    group_columns: list[str],
    top_quantiles: list[float],
    chunk_size: int = 4096,
) -> None:
    sentence_vectors = np.load(sentence_vectors_path, mmap_mode="r")
    phrase_vectors = np.load(phrase_vectors_path, mmap_mode="r")
    mapping = pd.read_csv(sentence_mapping_path)
    ensure_columns(mapping.columns, group_columns)

    if len(mapping) != sentence_vectors.shape[0]:
        raise ValueError("Sentence mapping row count does not match sentence vector count.")

    max_scores = _max_phrase_similarity(sentence_vectors, phrase_vectors, chunk_size)
    scored = mapping[group_columns].copy()
    scored["max_phrase_similarity"] = max_scores

    result = (
        scored.groupby(group_columns, dropna=False)["max_phrase_similarity"]
        .apply(lambda series: pd.Series(summarize_similarity(series.to_numpy(), top_quantiles)))
        .unstack()
        .reset_index()
    )
    save_csv(output_csv, result, "Saved similarity indicators")


def summarize_similarity(scores: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    result = {
        "similarity_mean": float(np.mean(scores)),
        "similarity_sum": float(np.sum(scores)),
        "similarity_max": float(np.max(scores)),
    }
    for quantile in quantiles:
        threshold = np.quantile(scores, quantile)
        kept = scores[scores >= threshold]
        label = str(int(quantile * 100)).zfill(2)
        result[f"similarity_sum_q{label}"] = float(np.sum(kept))
        result[f"similarity_mean_q{label}"] = float(np.mean(kept)) if len(kept) else 0.0
    return result


def _max_phrase_similarity(
    sentence_vectors: np.ndarray,
    phrase_vectors: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    max_scores = np.empty(sentence_vectors.shape[0], dtype=np.float32)
    for start in tqdm(range(0, sentence_vectors.shape[0], chunk_size), desc="Similarity"):
        end = min(start + chunk_size, sentence_vectors.shape[0])
        sim = cosine_similarity(sentence_vectors[start:end], phrase_vectors)
        max_scores[start:end] = sim.max(axis=1)
    return max_scores
