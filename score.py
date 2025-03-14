"""Compute the likelihood scores that an LLM assigns to particular sequences.
"""

import hydra
import json
import logging
import os
import pandas as pd
import pprint
import s3fs
import torch

from finetune_utils import formatting_texts_func_edit_pairs
from model_client import ModelClient
from omegaconf import DictConfig, OmegaConf


def score(cfg: DictConfig, logger: logging.Logger = None):
    model_client = ModelClient(
        model_name_or_path=cfg.model_name_or_path,
        logger=logger,
        max_generate_length=1,  # don't need to generate, only scoring
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    df = pd.read_json(cfg.data_path, orient="records", lines=True)
    if cfg.sanity_check:
        logger.warning(
            "Running in sanity check mode. Will reduce data down to 20 examples."
        )
        df = df.sample(n=20)
    data = df.to_dict("list")
    # inputs = df[cfg.input_field].to_list()
    # targets = df[cfg.target_field].to_list()
    formatted_inputs = formatting_texts_func_edit_pairs(
        data,
        include_target=False,
        higher_score_particle_field=cfg.input_field,
        lower_score_particle_field=cfg.target_field,
    )
    formatted_targets = [json.dumps(target) for target in data[cfg.target_field]]
    avg_likelihoods = model_client.compute_likelihoods(
        formatted_inputs,
        formatted_targets,
        batch_size=cfg.batch_size,
        logger=logger,
    )
    output_fp = os.path.join(cfg.output_dir, cfg.output_filename)
    data = df.to_dict("records")
    for avg_likelihood, record in zip(avg_likelihoods, data):
        record["likelihood"] = avg_likelihood
    output_df = pd.DataFrame(data)
    output_df.to_json(output_fp, orient="records", lines=True)


@hydra.main(config_path="config", config_name="score")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)
    score(cfg, logger)


if __name__ == "__main__":
    main()
