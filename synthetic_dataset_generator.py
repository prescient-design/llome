import copy
import hydra
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import synthetic_dataset_lib
import sys
import torch
import wandb
from botorch.test_functions import SyntheticTestFunction
from holo.logging import wandb_setup
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji
from itertools import product
from omegaconf import DictConfig, OmegaConf
from synthetic_dataset_lib import ranked_fft
from tqdm import tqdm
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def init_test_fns(
    cfg: DictConfig, device: torch.device, dtype: torch.dtype
) -> List[SyntheticTestFunction]:
    test_fns = []
    for _ in range(cfg.num_test_fns):
        # generate new seed for each test function
        seed = random.randint(0, 2**16)
        test_fn_cfg = cfg.test_function
        test_fn_cfg.random_seed = seed
        test_function = hydra.utils.instantiate(cfg.test_function)
        test_function = test_function.to(device, dtype)
        test_fns.append(test_function)
    return test_fns


def filter_sequences(
    particles_and_scores: List[Tuple[Tuple[int], float]], num_sequences: int
) -> List[Dict[str, Any]]:
    library = torch.stack(
        [torch.LongTensor(d["particle"]) for d in particles_and_scores]
    )
    ranking_scores = torch.tensor([d["score"] for d in particles_and_scores])
    filtered_idx = ranked_fft(
        library, ranking_scores, n=num_sequences, descending=False
    )
    filtered_particles_and_scores = [particles_and_scores[idx] for idx in filtered_idx]
    return filtered_particles_and_scores


def convert_test_fn_to_dict(test_fn: SyntheticTestFunction) -> Dict[str, Any]:
    if isinstance(test_fn, Ehrlich):
        motif_list = [motif.tolist() for motif in test_fn.motifs]
        spacing_list = [spacing.tolist() for spacing in test_fn.spacings]
        output_dict = {
            "num_states": test_fn.num_states,
            "dim": test_fn.dim,
            "num_motifs": len(test_fn.motifs),
            "motif_length": len(motif_list[0]),
            "motifs": motif_list,
            "spacings": spacing_list,
            "quantization": test_fn._quantization,
            "noise_std": test_fn.noise_std,
            "negate": test_fn.negate,
            "random_seed": test_fn._random_seed,
        }
    elif isinstance(test_fn, RoughMtFuji):
        output_dict = {
            "dim": test_fn.dim,
            "additive_term": test_fn._additive_term,
            "random_term_std": test_fn._random_term_std,
            "noise_std": test_fn.noise_std,
            "negate": test_fn.negate,
            "random_seed": test_fn._random_seed,
        }
    else:
        raise ValueError(f"Unknown test function type: {type(test_fn)}")
    return output_dict


def train_optimizer(
    cfg: DictConfig, test_function: SyntheticTestFunction, dtype: torch.dtype
) -> Set[Tuple[Tuple[int], float]]:
    """Trains the optimizer on the given test_function and returns a history of solutions and scores."""
    initial_solution = test_function.initial_solution().to(dtype)
    params = [torch.nn.Parameter(initial_solution)]
    np.random.seed(test_function._random_seed)
    torch.manual_seed(test_function._random_seed)

    logger.info(f"Test function: {test_function}")
    logger.info(f"Known optimal solution: {test_function.optimal_solution()}")

    def closure(param_list):
        return test_function(param_list[0])

    vocab = list(range(test_function.num_states))
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=params, vocab=vocab)

    logger.info(
        f"Searching for solution with optimal value {test_function.optimal_value}..."
    )
    cumulative_regret = 0.0
    best_loss = float("inf")
    all_particles_and_scores = {}  # dict of particle -> {"step": step, "score": score}
    num_particles_seen = 0
    for t_idx in tqdm(range(cfg.num_opt_steps)):
        loss = optimizer.step(closure)
        # TODO: Get the new particles!
        particles = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                particles.extend(state["particles"])
        # recompute scores because the ones stored are for the last iteration
        particle_scores = test_function(torch.stack(particles))
        assert len(particles) == len(particle_scores)
        for particle, score in zip(particles, particle_scores):
            # convert particle to tuple of ints to make it hashable!
            particle_tuple = tuple(int(x) for x in particle)
            # only add to history if this particle wasn't already found
            if particle_tuple not in all_particles_and_scores:
                all_particles_and_scores[particle_tuple] = {
                    "step": t_idx,
                    "score": score.item(),
                    "num_particles_seen": num_particles_seen,
                }
            num_particles_seen += 1
        logging.info(
            f"{len(all_particles_and_scores)} total particles stored at iter {t_idx}."
        )

        if loss < best_loss:
            best_loss = loss.item()
            best_params = [p.data.clone() for p in params]

        simple_regret_best = best_loss - test_function.optimal_value
        simple_regret_last = loss - test_function.optimal_value
        cumulative_regret += best_loss - test_function.optimal_value
        frac_particles_feasible = (
            optimizer.particle_loss.lt(float("inf")).float().mean().item()
        )

        metrics = {
            "simple_regret_best": simple_regret_best,
            "simple_regret_last": simple_regret_last,
            "cumulative_regret": cumulative_regret,
            "frac_particles_feasible": frac_particles_feasible,
            "timestep": t_idx,
        }

        stop = simple_regret_best == 0
        if t_idx % cfg.log_interval == 0 or stop:
            wandb.log(metrics)
            logger.info(f"Step {t_idx}: Loss {loss}")

        if stop:
            break
    logger.info(f"Best solution: {best_params[0].long()}")
    return all_particles_and_scores


@hydra.main(config_path="config", config_name="evol_dataset_gen")
def main(cfg: DictConfig):
    if cfg.optimizer.mutation_prob is None:
        cfg.optimizer.mutation_prob = 1.1 / cfg.test_function.dim
    if cfg.optimizer.recombine_prob is None:
        cfg.optimizer.recombine_prob = 1.1 / cfg.test_function.dim
    wandb_setup(cfg)
    random.seed(cfg.random_seed)
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(cfg.log_level.upper())

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    dtype = torch.double if cfg.dtype == "float64" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_fns = init_test_fns(cfg, device, dtype)

    hps = [
        [{hp_name: v} for v in hp_values]
        for hp_name, hp_values in cfg.hyperparameter_ranges.items()
    ]
    hp_combos = list(product(*hps))

    opt_cfg = cfg.optimizer
    all_particles_and_scores = {}
    all_cfgs = []
    for test_fn in test_fns:
        for hp_combo in hp_combos:
            curr_cfg = copy.deepcopy(opt_cfg)
            for param_dict in hp_combo:
                for param_name, param_val in param_dict.items():
                    if not hasattr(curr_cfg, param_name):
                        logger.warning(
                            f"Config does not have hyperparameter named {param_name}. Skipping..."
                        )
                        continue
                    setattr(curr_cfg, param_name, param_val)

            combo_cfg = copy.deepcopy(cfg)
            combo_cfg.optimizer = curr_cfg
            logger.info(f"Current full config: {combo_cfg}")
            all_cfgs.append(combo_cfg)
            curr_particles_and_scores = train_optimizer(combo_cfg, test_fn, dtype)
            all_particles_and_scores = {
                **all_particles_and_scores,
                **curr_particles_and_scores,
            }
            logging.info(f"{len(all_particles_and_scores)} particle and score pairs.")

    all_particles_and_scores = [
        {"particle": list(particle), **particle_dict}
        for particle, particle_dict in all_particles_and_scores.items()
    ]
    if cfg.max_sequences is not None:
        all_particles_and_scores = filter_sequences(
            all_particles_and_scores, num_sequences=cfg.max_sequences
        )
    # Now format and write to file
    plain_pairs_fp = os.path.join(cfg.output_dir, "plain_pairs.jsonl")
    plain_pairs = synthetic_dataset_lib.format_plain(all_particles_and_scores)
    plain_pairs_df = pd.DataFrame(plain_pairs)
    logging.info(f"Writing {len(plain_pairs_df)} rows to {plain_pairs_fp}.")
    plain_pairs_df.to_json(plain_pairs_fp, orient="records", lines=True)

    # also write out test functions and config to file
    test_fn_name = "ehrlich.jsonl" if isinstance(test_fn, Ehrlich) else "mt_fuji.jsonl"
    test_fn_fp = os.path.join(cfg.output_dir, test_fn_name)
    test_fn_dicts = [convert_test_fn_to_dict(test_fn) for test_fn in test_fns]
    test_fns_df = pd.DataFrame(test_fn_dicts)
    test_fns_df.to_json(test_fn_fp, orient="records", lines=True)

    cfg_df = []
    for combo_cfg in all_cfgs:
        cfg_df.append(OmegaConf.to_container(combo_cfg))
    cfg_df = pd.DataFrame(cfg_df)
    cfg_fp = os.path.join(cfg.output_dir, "cfgs.jsonl")
    cfg_df.to_json(cfg_fp, orient="records", lines=True)


if __name__ == "__main__":
    main()
