import datasets
import hydra
import logging
import os
import pandas as pd
import pprint
import torch

from botorch.test_functions import SyntheticTestFunction
from finetune_utils import (
    formatting_texts_func_edit_pairs,
    parse_particle_and_score,
    truncate_after_right_bracket_w_logps,
)
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji
from model_client import ModelClient
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import distance
from tqdm import tqdm
from transformers import GenerationConfig


def run_iterative_generation(cfg: DictConfig, logger: logging.Logger = None):
    test_fn_params = pd.read_json(cfg.test_fn_fp, orient="records", lines=True).to_dict(
        "records"
    )[0]
    if cfg.test_fn_type == "ehrlich":
        test_fn = Ehrlich(
            num_states=test_fn_params["num_states"],
            dim=test_fn_params["dim"],
            num_motifs=test_fn_params["num_motifs"],
            motif_length=test_fn_params["motif_length"],
            quantization=test_fn_params["quantization"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    else:
        test_fn = RoughMtFuji(
            dim=test_fn_params["dim"],
            additive_term=test_fn_params["additive_term"],
            random_term_std=test_fn_params["random_term_std"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    gen_config = hydra.utils.instantiate(cfg.generation_config)
    model_client = ModelClient(
        model_name_or_path=cfg.model_name_or_path,
        logger=logger,
        max_generate_length=gen_config.max_new_tokens,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    df = pd.read_json(cfg.data_path, orient="records", lines=True)
    if cfg.sanity_check:
        logger.info(
            f"Running in sanity check mode... reducing data down to 10 examples."
        )
        df = df.sample(n=min(10, len(df)))

    # Before sampling, save all the particles as tuples in a set so that we can check whether
    # generations are regurgitations from the training data
    dataset_particles = set(
        [tuple(ex[cfg.higher_score_particle_field]) for _, ex in df.iterrows()]
    ).union(set([tuple(ex[cfg.lower_score_particle_field]) for _, ex in df.iterrows()]))
    # Now dedupe the lower_score_particles and sample the lowest scoring examples from the data
    # to use as seeds for generation
    # df = df.drop_duplicates(subset=[cfg.lower_score_particle_field])
    if cfg.sampling_method == "best_scoring":
        df = df.sort_values(by=[cfg.lower_score_field], ascending=True)[
            : cfg.sample_size
        ]
    elif cfg.sampling_method == "uniform":
        df = df.sample(n=min(len(df), cfg.sample_size), random_state=cfg.seed)
    elif cfg.sampling_method == "combination":
        half_sample_size = int(cfg.sample_size / 2)
        df = pd.concat(
            [
                df.sort_values(by=[cfg.lower_score_field], ascending=True)[
                    :half_sample_size
                ],
                df.sample(n=min(len(df), half_sample_size), random_state=cfg.seed),
            ]
        )
    else:
        raise ValueError(f"Unknown sampling method '{cfg.sampling_method}.'")
    # Start by using the lower score particle from each pair as the seed
    ds = datasets.Dataset.from_pandas(df)
    input_ds = datasets.Dataset.from_list(
        [
            {
                cfg.higher_score_particle_field: ex[cfg.lower_score_particle_field],
                "score": ex[cfg.lower_score_field],
            }
            for ex in ds
        ]
    )
    all_trajectories = [
        [
            {
                "particle": ex[cfg.lower_score_particle_field],
                "score": ex[cfg.lower_score_field],
                "input_particle": None,
                "input_score": None,
                "seed_score": ex[cfg.lower_score_field],
                "in_dataset": True,
                "iter": 0,
                "loglikelihood": None,
                "num_particles_generated": 0,
                "hamming_distance": None,
            },
        ]
        for ex in ds
    ]
    num_particles_generated = 0
    for iter in tqdm(range(1, cfg.max_iterations + 1), desc="Generation iterations..."):
        input_texts = formatting_texts_func_edit_pairs(
            input_ds,
            include_target=False,
            higher_score_particle_field=cfg.higher_score_particle_field,
            lower_score_particle_field=cfg.lower_score_particle_field,
        )
        _, output_token_ids, output_token_logps = model_client.generate_texts_batched(
            input_texts,
            batch_size=cfg.batch_size,
            generation_config=gen_config,
            return_likelihoods=True,
        )
        trunc_outputs = []
        trunc_output_logps = []
        for token_ids, token_logps in tqdm(
            zip(output_token_ids, output_token_logps), desc="Truncating outputs.."
        ):
            trunc_output, logps = truncate_after_right_bracket_w_logps(
                token_ids, token_logps, model_client.tokenizer, length_normalized=True
            )
            trunc_outputs.append(trunc_output)
            trunc_output_logps.append(logps)
        logger.info(
            f"Len of trunc_outputs: {len(trunc_outputs)}\nLen of trunc_output_logps: {len(trunc_output_logps)}"
        )
        # store outputs and create inputs for the next iteration
        prev_input_ds = input_ds
        input_ds = []
        for trajectory_idx in range(len(all_trajectories)):
            for output_idx in range(gen_config.num_return_sequences):
                output = trunc_outputs[
                    trajectory_idx * gen_config.num_return_sequences + output_idx
                ]
                output_logp = trunc_output_logps[
                    trajectory_idx * gen_config.num_return_sequences + output_idx
                ]
                output_particle_and_score = parse_particle_and_score(output, test_fn)
                num_particles_generated += 1
                if output_particle_and_score is None:
                    continue
                input_particle = prev_input_ds[trajectory_idx][
                    cfg.higher_score_particle_field
                ]
                hamming_dist = distance.hamming(
                    input_particle, output_particle_and_score[0]
                )
                # If any of the outputs is parsable, then we continue to iteratively
                # generate for that example.
                all_trajectories[trajectory_idx].append(
                    {
                        "particle": output_particle_and_score[0],
                        "score": output_particle_and_score[1],
                        "input_particle": input_particle,
                        "input_score": prev_input_ds[trajectory_idx]["score"],
                        "seed_score": all_trajectories[trajectory_idx][0]["seed_score"],
                        "in_dataset": tuple(output_particle_and_score[0])
                        in dataset_particles,
                        "iter": iter,
                        "loglikelihood": output_logp,
                        "num_particles_generated": num_particles_generated,
                        "hamming_distance": hamming_dist,
                    }
                )
            # Only include the highest-likelihood output in the pool for a given example
            # in the inputs for the next round. If no particles have non-NaN log-likelihood, then
            # use the original seed.
            candidates = [
                d
                for d in all_trajectories[trajectory_idx]
                if d["loglikelihood"] is not None
            ]
            if len(candidates) > 0:
                max_likelihood_gen = max(candidates, key=lambda d: d["loglikelihood"])
            else:
                max_likelihood_gen = all_trajectories[trajectory_idx][0]
            input_ds.append(
                {
                    cfg.higher_score_particle_field: max_likelihood_gen["particle"],
                    "score": max_likelihood_gen["score"],
                }
            )

        input_ds = datasets.Dataset.from_list(input_ds)
    # Give each trajectory an ID and flatten out the list of outputs!
    all_trajectories = [
        {"trajectory_id": example_id, **d}
        for example_id, trajectory in enumerate(all_trajectories)
        for d in trajectory
    ]
    all_trajectories = pd.DataFrame(all_trajectories)
    all_trajectories.to_json(
        os.path.join(cfg.output_dir, cfg.output_filename), orient="records", lines=True
    )


@hydra.main(config_path="config", config_name="iterative_generation")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)
    run_iterative_generation(cfg, logger)


if __name__ == "__main__":
    main()
