"""
Given a set of (particle, score), generate a dataset for a particular format.
"""

import hydra
import json
import logging
import numpy as np
import pandas as pd
import pprint
import random
import torch

from omegaconf import DictConfig, OmegaConf
from pynndescent import PyNNDescentTransformer
from scipy.spatial import distance
from synthetic_dataset_lib import ranked_fft
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level="INFO", force=True)


def find_minimal_edit_pairs(cfg: DictConfig, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Given a dataset of particles and scores, find minimal pairs that have different scores.
    """
    logging.info(f"Length of original df: {len(df)}")
    df = df.drop_duplicates(subset=["particle"])
    logging.info(f"Length of df after removing duplicates: {len(df)}")
    X = np.array([p for p in df["particle"]])
    scores = np.array([s for s in df["score"]])
    if cfg.score_lower_threshold is not None:
        X, scores = filter_by_score(X, scores, cfg.score_lower_threshold)
    pynn_transformer = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(X)
    transformed = pynn_transformer.transform(X)

    # # take a smaller sample if necessary
    # if cfg.n is not None:
    #     random.seed(cfg.seed)
    #     transformed = transformed[
    #         random.sample(
    #             range(transformed.shape[0]), k=min(2 * cfg.n, transformed.shape[0])
    #         ),
    #         :,
    #     ]

    # loop through rows instead of doing matrix computation to save memory
    idx_pairs = []
    for i in tqdm(range(transformed.shape[0])):
        # find elements that are nearest neighbors but have different score
        # from the original particle
        curr_score = scores[i]
        nearest_neighbor_idxs = set(transformed.getrow(i).nonzero()[1])
        diff_score_idxs = set(np.nonzero(scores != curr_score)[0])
        # now find the intersection
        eligible_elem_idxs = nearest_neighbor_idxs & diff_score_idxs
        if len(eligible_elem_idxs) == 0:
            logging.warning(f"No eligible neighbors found.")
            continue
        eligible_elem_idxs = list(eligible_elem_idxs)
        closest_eligible_nbor_idx = eligible_elem_idxs[
            np.argmin(
                np.array(
                    transformed.getrow(i)[
                        [0 for _ in range(len(eligible_elem_idxs))], eligible_elem_idxs
                    ]
                )[0]
            )
        ]
        idx_pairs.append((i, closest_eligible_nbor_idx))

    if cfg.n is not None:
        random.seed(cfg.seed)
        idx_pairs = random.sample(idx_pairs, k=min(cfg.n, len(idx_pairs)))
    outputs = get_outputs_from_idx_pairs(idx_pairs, X, scores)
    if cfg.max_proportion_infeasible is not None:
        outputs = filter_infeasible_examples(cfg, outputs)
    return outputs


def filter_by_score(
    library: torch.Tensor, scores: torch.Tensor, score_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    idxs = torch.where(scores > score_threshold)
    return library[idxs], scores[idxs]


def filter_by_score_df(df: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    return df[df["score"] > score_threshold]


def abs_subtract_replace_infs(x: float, y: float) -> float:
    """Replace the infinities with a score of 0.0."""
    if np.isinf(x) or np.isnan(x):
        x = 0.0
    if np.isinf(y) or np.isnan(y):
        y = 0.0
    return np.abs(x - y)


def filter_by_likelihood_range(
    df: pd.DataFrame, likelihood_quantile_range: Tuple[float, float]
) -> pd.DataFrame:
    orig_size = len(df)
    if "likelihood" not in df.columns and "loglikelihood" not in df.columns:
        logging.error(
            f"Neither 'likelihood' nor 'loglikelihood' are in DF columns. Not filtering by likelihood range."
        )
        return df
    elif "likelihood" not in df.columns:
        df["likelihood"] = df["loglikelihood"].map(lambda x: np.exp(x))
    lower_ll_bound = df["likelihood"].quantile(likelihood_quantile_range[0])
    upper_ll_bound = df["likelihood"].quantile(likelihood_quantile_range[1])
    df = df[(df["likelihood"] >= lower_ll_bound) & (df["likelihood"] <= upper_ll_bound)]
    logging.info(
        f"Filtered dataset from {orig_size} down to {len(df)} examples after filtering by likelihood range."
    )
    return df


def filter_by_likelihood_lower_threshold(
    df: pd.DataFrame, likelihood_quantile_threshold: float
) -> pd.DataFrame:
    orig_size = len(df)
    if "likelihood" not in df.columns and "loglikelihood" not in df.columns:
        logging.error(
            f"Neither 'likelihood' nor 'loglikelihood' are in DF columns. Not filtering by likelihood range."
        )
        return df
    elif "likelihood" not in df.columns:
        df["likelihood"] = df["loglikelihood"].map(lambda x: np.exp(x))
    likelihood_lower_threshold = df["likelihood"].quantile(
        likelihood_quantile_threshold
    )
    df = df[(df["likelihood"] >= likelihood_lower_threshold)]
    logging.info(
        f"Filtered dataset from {orig_size} down to {len(df)} examples after filtering by lower likelihood threshold."
    )
    return df


def find_dense_pairs(
    cfg: DictConfig, df: pd.DataFrame, allow_same_score_pair: bool = False
) -> List[Dict[str, Any]]:
    """Adapted from PropEn -- find all pairs within a specific edit distance of each other"""
    df = df.drop_duplicates(subset=["particle"])
    if cfg.n is not None:
        if cfg.filter_by_likelihood and len(df) > cfg.n:
            df = filter_by_likelihood_lower_threshold(
                df, cfg.likelihood_quantile_threshold
            )
        elif cfg.filter_by_likelihood_range and len(df) > cfg.n:
            df = filter_by_likelihood_range(df, tuple(cfg.likelihood_quantile_range))
        df = df.sample(n=min(len(df), cfg.n), random_state=cfg.seed)
    if cfg.score_lower_threshold is not None:
        df = filter_by_score_df(df, cfg.score_lower_threshold)
    library = torch.stack([torch.LongTensor(p) for p in df["particle"]])
    ranking_scores = torch.FloatTensor([x for x in df["score"]])
    filtered = library
    filtered_scores = ranking_scores
    filtered = filtered.numpy()
    pynn_transformer = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(filtered)
    transformed = pynn_transformer.transform(filtered)
    # loop through rows instead of doing matrix computation to save memory
    idx_pairs = set([])
    for i in tqdm(range(transformed.shape[0]), desc="Finding pairs"):
        # find elements that are nearest neighbors but have different score
        # from the original particle and are within a bounded distance
        row = transformed.getrow(i)
        nearest_neighbor_idxs = row.nonzero()[1]
        within_x_distance_idxs = [
            idx
            for idx, nn_dist in zip(
                nearest_neighbor_idxs,
                np.array(
                    row[
                        [0 for _ in range(nearest_neighbor_idxs.shape[-1])],
                        nearest_neighbor_idxs,
                    ]
                )[0].tolist(),
            )
            if nn_dist <= cfg.dist_x_threshold
        ]
        if cfg.dist_y_threshold is not None:
            selected_idxs = [
                idx
                for idx, score in zip(
                    within_x_distance_idxs, filtered_scores[within_x_distance_idxs]
                )
                if abs_subtract_replace_infs(score, filtered_scores[i])
                <= cfg.dist_y_threshold
            ]
        else:
            selected_idxs = within_x_distance_idxs
        for idx in selected_idxs:
            if i < idx:
                idx_pairs.add((i, idx))
            else:
                idx_pairs.add((idx, i))
    idx_pairs = list(idx_pairs)
    logging.info(f"{len(idx_pairs)} index pairs satisfied conditions.")
    outputs = get_outputs_from_idx_pairs(
        idx_pairs,
        filtered,
        filtered_scores,
        allow_same_score_pair=allow_same_score_pair,
        df=df,
    )
    if cfg.max_proportion_infeasible is not None:
        outputs = filter_infeasible_examples(cfg, outputs)
    return outputs


def get_score_pairs_df(
    i: int,
    j: int,
    scores: torch.Tensor,
    library: torch.Tensor,
    allow_same_score_pair: bool = False,
    df: pd.DataFrame = None,
) -> Optional[Dict[str, Any]]:
    score_i = scores[i]
    score_j = scores[j]
    particle_i = library[i]
    particle_j = library[j]
    if torch.isnan(score_i).item():
        score_i = torch.inf
    if torch.isnan(score_j).item():
        score_j = torch.inf
    if score_i > score_j:
        higher_score = score_i
        lower_score = score_j
        higher_score_p = particle_i
        lower_score_p = particle_j
        higher_idx = i
        lower_idx = j
    elif score_j > score_i:
        higher_score = score_j
        lower_score = score_i
        higher_score_p = particle_j
        lower_score_p = particle_i
        higher_idx = j
        lower_idx = i
    elif not allow_same_score_pair:
        logging.warning(
            f"Pair of particles with the same score:\n{particle_i}\n{particle_j}\nScore: {score_i}"
        )
        return None
    else:
        higher_score = score_i
        lower_score = score_j
        higher_score_p = particle_i
        lower_score_p = particle_j
        higher_idx = i
        lower_idx = j

    output_dict = {
        "lower_score_particle": lower_score_p.tolist(),
        "lower_score": f"{lower_score:.3f}",
        "higher_score_particle": higher_score_p.tolist(),
        "higher_score": f"{higher_score:.3f}",  # higher score is worse! can be inf
    }
    if "loglikelihood" in df.columns:
        output_dict["lower_particle_loglikelihood"] = df.iloc[lower_idx][
            "loglikelihood"
        ]
        output_dict["higher_particle_loglikelihood"] = df.iloc[higher_idx][
            "loglikelihood"
        ]
    return output_dict


def get_score_pairs(
    score_i: torch.Tensor,  # single-element tensor
    score_j: torch.Tensor,
    particle_i: torch.Tensor,
    particle_j: torch.Tensor,
    allow_same_score_pair: bool = False,
) -> Optional[Dict[str, Any]]:
    if torch.isnan(score_i).item():
        score_i = torch.inf
    if torch.isnan(score_j).item():
        score_j = torch.inf
    if score_i > score_j:
        higher_score = score_i
        lower_score = score_j
        higher_score_p = particle_i
        lower_score_p = particle_j
    elif score_j > score_i:
        higher_score = score_j
        lower_score = score_i
        higher_score_p = particle_j
        lower_score_p = particle_i
    elif not allow_same_score_pair:
        logging.warning(
            f"Pair of particles with the same score:\n{particle_i}\n{particle_j}\nScore: {score_i}"
        )
        return None
    else:
        higher_score = score_i
        lower_score = score_j
        higher_score_p = particle_i
        lower_score_p = particle_j

    output_dict = {
        "lower_score_particle": lower_score_p.tolist(),
        "lower_score": f"{lower_score:.3f}",
        "higher_score_particle": higher_score_p.tolist(),
        "higher_score": f"{higher_score:.3f}",  # higher score is worse! can be inf
    }
    return output_dict


def get_outputs_from_idx_pairs(
    idx_pairs: List[Tuple[int, int]],
    library: torch.Tensor,
    scores: torch.Tensor,
    allow_same_score_pair: bool = False,
    df: pd.DataFrame = None,
) -> List[Dict[str, Any]]:
    outputs = []
    for i, j in tqdm(idx_pairs, desc="Creating output records from index pairs"):
        output_dict = get_score_pairs_df(
            i, j, scores, library, allow_same_score_pair=allow_same_score_pair, df=df
        )
        if output_dict is not None:
            outputs.append(output_dict)
    logging.info(f"{len(outputs)} output records.")
    return outputs


def filter_infeasible_examples(
    cfg: DictConfig,
    curr_examples: List[Dict[str, Any]],
    score_field: str = "higher_score",
) -> List[Dict[str, Any]]:
    no_infeasible_seq = [
        i for i, ex in enumerate(curr_examples) if ex[score_field] != "inf"
    ]
    w_infeasible_seq = [
        i for i, ex in enumerate(curr_examples) if ex[score_field] == "inf"
    ]
    num_feasible = len(no_infeasible_seq)
    max_infeasible_examples = (
        num_feasible
        * cfg.max_proportion_infeasible
        / (1 - cfg.max_proportion_infeasible)
    )
    max_infeasible_examples = int(np.round(max_infeasible_examples))
    random.seed(cfg.seed)
    w_infeasible_seq = random.sample(
        w_infeasible_seq, k=min(len(w_infeasible_seq), max_infeasible_examples)
    )
    data_idxs = no_infeasible_seq + w_infeasible_seq
    random.shuffle(data_idxs)
    output_data = [curr_examples[i] for i in data_idxs]
    logging.info(
        f"Filtered dataset from {len(curr_examples)} to {len(output_data)} examples after downsampling infeasible examples."
    )
    return output_data


def find_preference_pairs(cfg: DictConfig, df: pd.DataFrame) -> List[Dict[str, Any]]:
    random.seed(cfg.seed)
    df = df.drop_duplicates(subset=["particle"])
    if cfg.n is not None:
        if cfg.filter_by_likelihood and len(df) > cfg.n:
            df = filter_by_likelihood_lower_threshold(
                df, cfg.likelihood_quantile_threshold
            )
        elif cfg.filter_by_likelihood_range and len(df) > cfg.n:
            df = filter_by_likelihood_range(df, tuple(cfg.likelihood_quantile_range))
        df = df.sample(n=min(len(df), cfg.n), random_state=cfg.seed)

    if cfg.score_lower_threshold is not None:
        df = filter_by_score_df(df, cfg.score_lower_threshold)
    if cfg.max_proportion_infeasible is not None:
        data = filter_infeasible_examples(
            cfg, df.to_dict("records"), score_field="score"
        )
    else:
        data = df.to_dict("records")
    library = torch.stack([torch.LongTensor(p["particle"]) for p in data])
    ranking_scores = torch.FloatTensor([x["score"] for x in data])
    filtered = library
    filtered_scores = ranking_scores
    filtered = filtered.numpy()
    pynn_transformer = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(filtered)
    transformed = pynn_transformer.transform(filtered)
    # loop through rows instead of doing matrix computation to save memory
    idx_triples = set([])
    for i in tqdm(range(transformed.shape[0]), desc="Finding pairs"):
        # find elements that are nearest neighbors
        # and are within a bounded distance
        row = transformed.getrow(i)
        nearest_neighbor_idxs = row.nonzero()[1]
        within_x_distance_idxs = [
            idx
            for idx, nn_dist in zip(
                nearest_neighbor_idxs,
                np.array(
                    row[
                        [0 for _ in range(nearest_neighbor_idxs.shape[-1])],
                        nearest_neighbor_idxs,
                    ]
                )[0].tolist(),
            )
            if nn_dist <= cfg.dist_x_threshold
        ]
        # get all possible pairs of (y1, y2)
        all_pair_idxs = [
            (k, j) for k in within_x_distance_idxs for j in within_x_distance_idxs
        ]
        # also add the index i itself in pairs
        all_pair_idxs.extend([(i, k) for k in within_x_distance_idxs])
        curr_score = filtered_scores[i].item()
        # loop through all possible triples of (x, y1, y2) and check if valid
        for pair_idx in all_pair_idxs:
            first_score = filtered_scores[pair_idx[0]].item()
            second_score = filtered_scores[pair_idx[1]].item()
            if first_score < curr_score and second_score >= curr_score:
                # y1 is yw, y2 is yl
                idx_triples.add((i, pair_idx[0], pair_idx[1]))
            elif second_score < curr_score and first_score >= curr_score:
                idx_triples.add((i, pair_idx[1], pair_idx[0]))
    idx_triples = list(idx_triples)
    outputs = []
    for x_idx, yw_idx, yl_idx in idx_triples:
        output_dict = {
            "prompt": filtered[x_idx].tolist(),
            "prompt_score": f"{filtered_scores[x_idx]:.3f}",
            "chosen": filtered[yw_idx].tolist(),
            "chosen_score": f"{filtered_scores[yw_idx]:.3f}",
            "rejected": filtered[yl_idx].tolist(),
            "rejected_score": f"{filtered_scores[yl_idx]:.3f}",
        }
        if "loglikelihood" in data[0].keys():
            output_dict["prompt_loglikelihood"] = data[x_idx]["loglikelihood"]
            output_dict["chosen_loglikelihood"] = data[yw_idx]["loglikelihood"]
            output_dict["rejected_loglikelihood"] = data[yl_idx]["loglikelihood"]
        outputs.append(output_dict)
    return outputs


@hydra.main(config_path="config", config_name="dataset_edit_pairs")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(f"Config:\n{pprint.pformat(OmegaConf.to_container(cfg))}")
    df = pd.read_json(cfg.source_dataset_path, orient="records", lines=True)
    df["particle"] = df["particle"].map(
        lambda input_str: (
            [int(x) for x in json.loads(input_str)]
            if isinstance(input_str, str)
            else [int(x) for x in input_str]
        )
    )
    if cfg.format == "minimal_edit_pairs":
        outputs = find_minimal_edit_pairs(cfg, df)
    elif cfg.format == "dense_neighborhood_pairs":
        outputs = find_dense_pairs(
            cfg, df, allow_same_score_pair=cfg.allow_same_score_pair
        )
    elif cfg.format == "dense_preference_pairs":
        outputs = find_preference_pairs(cfg, df)
    else:
        raise ValueError(f"Unknown format: '{cfg.format}'.")
    outputs = pd.DataFrame(outputs)
    outputs.to_json(cfg.output_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
