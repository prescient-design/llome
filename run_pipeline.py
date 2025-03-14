import hydra
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import sys

from file_handler import LocalOrS3Client
from omegaconf import DictConfig, OmegaConf
from slurm_utils import submit_cmd_to_slurm, wait_for_slurm_jobs_to_complete
from tqdm import tqdm
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


def get_all_strs_from_nested_dict(nested_dict: Dict[str, Any]) -> List[str]:
    outputs = []
    for k, v in nested_dict.items():
        if not isinstance(v, Mapping):
            outputs.append(f"{k}={v}")
        else:
            for o in get_all_strs_from_nested_dict(v):
                outputs.append(f"{k}.{o}")
    return outputs


def generate_ga_dataset(cfg: DictConfig, fs: LocalOrS3Client) -> str:
    """
    Either run genetic algorithm to generate dataset, or simply return directory containing data.
    """
    python_cmd_str = "python -m synthetic_dataset_generator "
    opts = get_all_strs_from_nested_dict(cfg["evol_dataset_gen"]["args"])
    opts_str = " ".join(opts)
    python_cmd_str += f"{opts_str} "
    num_motifs = cfg["evol_dataset_gen"]["args"]["test_function"]["num_motifs"]
    motif_length = cfg["evol_dataset_gen"]["args"]["test_function"]["motif_length"]
    dim = cfg["evol_dataset_gen"]["args"]["test_function"]["dim"]
    num_particles = cfg["evol_dataset_gen"]["args"]["optimizer"]["num_particles"]
    mutation_prob = cfg["evol_dataset_gen"]["args"]["optimizer"]["mutation_prob"]
    num_opt_steps = cfg["evol_dataset_gen"]["args"]["num_opt_steps"]
    ds_name = f"c{num_motifs}_k{motif_length}_l{dim}_n{num_particles}_pm{mutation_prob}_steps{num_opt_steps}"
    output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{ds_name}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{ds_name}"
    )
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)
    python_cmd_str += f"output_dir={output_dir} "
    output_fp = f"{output_dir}/plain_pairs.jsonl"
    if not cfg.overwrite and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_dir
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_evol_dataset_gen:
        slurm_kwargs = OmegaConf.to_container(cfg.evol_dataset_gen.slurm_args)
        slurm_kwargs["job_name"] = "ga_seeds"
        submit_cmd_to_slurm(
            python_cmd_str, cfg.conda_env, slurm_dump_dir, blocking=True, **slurm_kwargs
        )

    return output_dir


def create_propen_sft_dataset(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    source_dataset_fp: str,
    filename_prefix: str = "",
    n: int = None,
    **extra_kwargs,
) -> str:
    python_cmd_str = "python -m synthetic_dataset_formatter "
    opts = get_all_strs_from_nested_dict(cfg["propen_dataset_formatting_sft"]["args"])
    opts_str = " ".join(opts)
    opts_str += (
        f" source_dataset_path={source_dataset_fp} format=dense_neighborhood_pairs "
    )
    pdf_cfg = cfg.propen_dataset_formatting_sft.args
    output_fn = f"{filename_prefix}dense_neighborhood_pairs_xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"
    output_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{output_fn}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{output_fn}"
    )
    if not output_fp.startswith("s3://"):
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    opts_str += f"output_path={output_fp} "
    for k, v in extra_kwargs.items():
        opts_str += f"{k}={v} "
    if n is not None:
        opts_str += f"n={n} "
    python_cmd_str += f"{opts_str} "
    if not cfg.overwrite and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_propen_sft_dataset_formatting:
        slurm_kwargs = OmegaConf.to_container(
            cfg.propen_dataset_formatting_sft.slurm_args
        )
        slurm_kwargs["job_name"] = "propen_sft_formatting"
        submit_cmd_to_slurm(
            python_cmd_str, cfg.conda_env, slurm_dump_dir, blocking=True, **slurm_kwargs
        )
    return output_fp


def create_propen_preference_dataset(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    source_dataset_fp: str,
    filename_prefix: str = "",
    n: int = None,
) -> str:
    python_cmd_str = "python -m synthetic_dataset_formatter "
    opts = get_all_strs_from_nested_dict(
        cfg["propen_dataset_formatting_preference"]["args"]
    )
    opts_str = " ".join(opts)
    opts_str += (
        f" source_dataset_path={source_dataset_fp} format=dense_preference_pairs "
    )
    pdf_cfg = cfg.propen_dataset_formatting_preference.args
    output_fn = f"{filename_prefix}xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"
    output_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{output_fn}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{output_fn}"
    )
    if not output_fp.startswith("s3://"):
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    opts_str += f"output_path={output_fp} "
    if n is not None:
        opts_str += f"n={n} "

    python_cmd_str += f"{opts_str} "
    if not cfg.overwrite and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_propen_dpo_dataset_formatting:
        slurm_kwargs = OmegaConf.to_container(
            cfg.propen_dataset_formatting_preference.slurm_args
        )
        slurm_kwargs["job_name"] = "propen_dpo_formatting"
        submit_cmd_to_slurm(
            python_cmd_str, cfg.conda_env, slurm_dump_dir, blocking=True, **slurm_kwargs
        )
    return output_fp


def train_sft(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,
    ga_data_dir: str,
    sft_run_name: str,
    model_dir: str = "EleutherAI/pythia-2.8b",
    train_from_scratch: bool = False,
) -> str:
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"
    os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{sft_run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{sft_run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )
    args = f"--config-name=pythia-2.8b_edit_pairs data_fp={data_fp} "
    args += " ".join(get_all_strs_from_nested_dict(cfg["sft"]["args"])) + " "
    args += f"test_fn_type=ehrlich test_fn_fp={test_fn_fp} "
    args += f"job_name={sft_run_name} s3_output_dir={s3_output_dir} "
    args += f"model_config.model_name_or_path={model_dir} "
    args += f"sanity_check={cfg.sanity_check} "

    # train from scratch
    if train_from_scratch and hasattr(cfg, "initial_model_config"):
        args += f"train_from_scratch=True "
        for k, v in cfg.initial_model_config.items():
            args += f"+init_model_config.{k}={v} "

    model_index_fp = (
        f"{s3_output_dir}/model.safetensors.index.json"
        if cfg.parent_output_dir is not None
        else f"{output_dir}/model.safetensors.index.json"
    )
    if not cfg.overwrite and fs.exists(model_index_fp):
        logger.info(f"{model_index_fp} already exists. Skipping...")
        return os.path.dirname(model_index_fp)
    else:
        logger.info(f"Did not find {model_index_fp}. Continuing to train...")
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_sft:
        slurm_cfg = cfg.sft.slurm_args
        # run with ddp (TODO: switch to fsdp)
        py_cmd = f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={slurm_cfg.gpus_per_node} "
        py_cmd += f"-m finetune_ehrlich {args} training_args.output_dir={output_dir}\n"

        # store return code for the finetuning job so that we can return it later
        py_cmd += f"RETURN_CODE=$?\n"

        # add extra commands for deleting local checkpoints after the job finishes
        # if S3 was used
        if cfg.parent_output_dir is not None:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

        # return the exit code of the finetuning job
        py_cmd += "exit ${RETURN_CODE}\n"

        slurm_kwargs = OmegaConf.to_container(cfg.sft.slurm_args)
        slurm_kwargs["job_name"] = "sft"
        submit_cmd_to_slurm(
            py_cmd, cfg.conda_env, slurm_dump_dir, blocking=True, **slurm_kwargs
        )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path


def train_dpo(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    ref_model_path: str,
    data_fp: str,
    ga_data_dir: str,
    run_name: str,
) -> str:
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )
    model_index_fp = (
        f"{s3_output_dir}/model.safetensors.index.json"
        if cfg.parent_output_dir is not None
        else f"{output_dir}/model.safetensors.index.json"
    )
    if not cfg.overwrite and fs.exists(model_index_fp):
        logger.info(f"{model_index_fp} already exists. Skipping...")
        return os.path.dirname(model_index_fp)
    args = f"--config-name=pythia-2.8b-dpo "
    args += " ".join(get_all_strs_from_nested_dict(cfg["dpo"]["args"])) + " "
    args += f"data_fp={data_fp} "
    args += f"dpo_config.run_name={run_name} "
    args += f"model_config.model_name_or_path={ref_model_path} "
    args += f"test_fn_fp={test_fn_fp} "
    args += f"job_name={run_name} "
    args += f"s3_output_dir={s3_output_dir} "
    args += f"dpo_script_args.sanity_check={cfg.sanity_check} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    slurm_cfg = cfg.dpo.slurm_args
    # run with ddp (TODO: switch to fsdp)
    py_cmd = f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={slurm_cfg.gpus_per_node} "
    py_cmd += f"-m dpo {args} dpo_config.output_dir={output_dir}\n"

    # store return code for the training job so that we can return it later
    py_cmd += f"RETURN_CODE=$?\n"

    # add extra commands for deleting local checkpoints after the job finishes
    # since they will have already been transferred to S3
    if cfg.parent_output_dir is not None:
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

    # return the exit code of the training job
    py_cmd += "exit ${RETURN_CODE}\n"

    slurm_kwargs = OmegaConf.to_container(cfg.dpo.slurm_args)
    slurm_kwargs["job_name"] = "dpo"
    submit_cmd_to_slurm(
        py_cmd, cfg.conda_env, slurm_dump_dir, blocking=True, **slurm_kwargs
    )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path


def train_marge(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    ref_model_path: str,
    data_fp: str,
    ga_data_dir: str,
    run_name: str,
) -> str:
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )
    model_index_fp = (
        f"{s3_output_dir}/model.safetensors.index.json"
        if cfg.parent_output_dir is not None
        else f"{output_dir}/model.safetensors.index.json"
    )
    if not cfg.overwrite and fs.exists(model_index_fp):
        logger.info(f"{model_index_fp} already exists. Skipping...")
        return os.path.dirname(model_index_fp)
    args = f"--config-name=pythia-2.8b-marge "
    args += " ".join(get_all_strs_from_nested_dict(cfg["marge"]["args"])) + " "
    args += f"data_fp={data_fp} "
    args += f"marge_config.run_name={run_name} "
    args += f"model_config.model_name_or_path={ref_model_path} "
    args += f"test_fn_fp={test_fn_fp} "
    args += f"job_name={run_name} "
    args += f"s3_output_dir={s3_output_dir} "
    args += f"dpo_script_args.sanity_check={cfg.sanity_check} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    slurm_cfg = cfg.marge.slurm_args
    # run with ddp (TODO: switch to fsdp)
    py_cmd = f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={slurm_cfg.gpus_per_node} "
    py_cmd += f"-m marge {args} marge_config.output_dir={output_dir}\n"

    # store return code for the training job so that we can return it later
    py_cmd += f"RETURN_CODE=$?\n"

    # add extra commands for deleting local checkpoints after the job finishes
    # since they will have already been transferred to S3
    if cfg.parent_output_dir is not None:
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

    # return the exit code of the training job
    py_cmd += "exit ${RETURN_CODE}\n"

    slurm_kwargs = OmegaConf.to_container(cfg.marge.slurm_args)
    slurm_kwargs["job_name"] = "marge"
    submit_cmd_to_slurm(
        py_cmd, cfg.conda_env, slurm_dump_dir, blocking=True, **slurm_kwargs
    )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path


def combine_datasets(
    cfg: DictConfig, fs: LocalOrS3Client, input_fps: List[str], output_fp: str
):
    """Combines multiple JSONL files."""
    if not cfg.overwrite and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return
    combined_df = pd.concat(
        [pd.read_json(input_fp, orient="records", lines=True) for input_fp in input_fps]
    )
    if "hamming_distance" in combined_df.columns:
        hd_avg = combined_df[~np.isnan(combined_df["hamming_distance"])][
            "hamming_distance"
        ].mean()
        if np.isnan(hd_avg):
            hd_avg = None
        else:
            hd_avg = hd_avg.item()
    else:
        hd_avg = None
    combined_df = combined_df.drop_duplicates(subset=["particle"])
    combined_df.to_json(output_fp, orient="records", lines=True)
    logger.info(f"Datasets combined and written to {output_fp}.")
    return hd_avg


def combine_new_with_old_datasets(
    cfg: DictConfig, fs: LocalOrS3Client, old_fps: List[str], curr_fp: str
):
    """Combines the current dataset with some old data."""
    if cfg.proportion_of_old_data == 0.0 or len(old_fps) == 0:
        return curr_fp
    output_fp = ".".join(curr_fp.split(".")[:-1]) + "_combined.jsonl"
    if not cfg.overwrite and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp

    curr = pd.read_json(curr_fp, orient="records", lines=True)
    combined_datasets = [curr]
    num_curr_rows = len(curr)
    num_old_datasets = len(old_fps)
    num_rows_per_old_fp = int(
        math.ceil(
            num_curr_rows
            * cfg.proportion_of_old_data
            / (num_old_datasets * (1.0 - cfg.proportion_of_old_data))
        )
    )
    for old_fp in old_fps:
        df = pd.read_json(old_fp, orient="records", lines=True)
        df = df.sample(n=min(len(df), num_rows_per_old_fp), random_state=cfg.seed)
        combined_datasets.append(df)
    combined_datasets = pd.concat(combined_datasets)

    combined_datasets.to_json(output_fp, orient="records", lines=True)
    logger.info(f"Combined current dataset with old datasets. Written to {output_fp}.")
    return output_fp


def run_iterative_generation(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,
    data_dir: str,
    model_dir: str,
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    lower_score_field: str = "lower_score",
    higher_score_field: str = "higher_score",
    temps: List[float] = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
) -> str:
    """
    Runs iterative generation jobs, combines the outputs, and returns the combined output filepath.
    """
    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["iterative_generation"]["args"])
    )
    args = f"{opt_str} data_path={data_fp} model_name_or_path={model_dir} output_dir={model_dir} "
    args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
    args += f"higher_score_particle_field={higher_score_particle_field} "
    args += f"lower_score_particle_field={lower_score_particle_field} "
    args += f"lower_score_field={lower_score_field} "
    args += f"higher_score_field={higher_score_field} "
    args += f"sanity_check={cfg.sanity_check} "

    output_filename_prefix = f"gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"
    greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
    temp_sampling_gen_args = [
        f"generation_config.do_sample=True generation_config.num_beams=1 "
        + f"+generation_config.temperature={temp} "
        + f"generation_config.num_return_sequences={cfg.generation_sampling_num_return_sequences} "
        + f"batch_size={cfg.sampling_gen_batch_size} "
        for temp in temps
    ]
    all_gen_args = [greedy_decoding_gen_args, *temp_sampling_gen_args]
    output_filenames = [
        f"{output_filename_prefix}_greedy.jsonl",
        *[
            f"{output_filename_prefix}_temp{temp}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"
            for temp in temps
        ],
    ]
    output_filepaths = [f"{model_dir}/{output_fn}" for output_fn in output_filenames]
    combined_outputs_fp = f"{model_dir}/{output_filename_prefix}.jsonl"
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_iter_gen:
        all_args = []
        for gen_args, output_fn in zip(all_gen_args, output_filenames):
            if not cfg.overwrite and fs.exists(f"{model_dir}/{output_fn}"):
                logger.info(f"{model_dir}/{output_fn} already exists. Skipping...")
            else:
                all_args.append(f"{args} {gen_args} output_filename={output_fn}")
        all_python_commands = [f"python -m iterative_generation {a}" for a in all_args]
        slurm_kwargs = OmegaConf.to_container(cfg.iterative_generation.slurm_args)
        slurm_kwargs["job_name"] = "iter_gen"
        job_submissions = [
            submit_cmd_to_slurm(
                py_cmd, cfg.conda_env, slurm_dump_dir, blocking=False, **slurm_kwargs
            )
            for py_cmd in all_python_commands
        ]
        wait_for_slurm_jobs_to_complete(job_submissions)
        hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)
    return combined_outputs_fp, hd


def get_temperatures(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir: str,
    prev_hd: Optional[float],
) -> List[float]:
    # check if previous temperatures file already exists
    temps_fp = f"{model_dir}/temperatures.json"
    if fs.exists(temps_fp):
        logger.info(f"Loading generation temperatures from {temps_fp}.")
        temps = json.load(fs.open(temps_fp))
    else:
        # if not already existing, dynamically compute and write to file
        temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        if cfg.temperature_scaling and prev_hd is not None:
            if prev_hd < 0.075:
                temps = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
            elif prev_hd < 0.1:
                temps = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            elif prev_hd < 0.15:
                temps = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
        with fs.open(temps_fp, "w") as f:
            f.write(json.dumps(temps))
    return temps


@hydra.main(config_path="config", config_name="pipeline")
def main(cfg: DictConfig):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(cfg.log_level.upper())
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    use_s3 = cfg.parent_output_dir is not None
    file_client_args = {"init_s3": use_s3}
    if use_s3:
        file_client_args["anon"] = False
        file_client_args["use_listings_cache"] = False
    file_client = LocalOrS3Client(**file_client_args)

    # Save full config to file!
    cfg_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/config.json"
        if use_s3
        else f"{cfg.local_output_dir}/{cfg.run_name}/config.json"
    )
    if not use_s3:
        os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    if not cfg.overwrite and file_client.exists(cfg_fp):
        logger.info(f"{cfg_fp} already exists. Not overwiting.")
    else:
        cfg_dicts = pd.DataFrame([OmegaConf.to_container(cfg)])
        cfg_dicts.to_json(cfg_fp, orient="records", lines=True)

    ga_data_dir = generate_ga_dataset(cfg, file_client)
    logger.info(f"GA dataset dir: {ga_data_dir}")

    all_prev_sft_datasets = []
    prev_round_outputs_fp = f"{ga_data_dir}/plain_pairs.jsonl"
    curr_model = cfg.initial_model
    prev_hd = None
    for i in tqdm(range(cfg.num_sft_rounds), desc="SFT iterations"):
        n = cfg.num_labels_after_first_round if i > 0 else None
        sft_dataset_fp = create_propen_sft_dataset(
            cfg, file_client, prev_round_outputs_fp, filename_prefix=f"sft_r{i}_", n=n
        )
        combined_sft_dataset_fp = combine_new_with_old_datasets(
            cfg, file_client, all_prev_sft_datasets, sft_dataset_fp
        )
        logger.info(f"SFT dataset path: {combined_sft_dataset_fp}")
        all_prev_sft_datasets.append(sft_dataset_fp)

        train_from_scratch = curr_model == cfg.initial_model and hasattr(
            cfg, "initial_model_config"
        )
        sft_dir = train_sft(
            cfg,
            file_client,
            combined_sft_dataset_fp,
            ga_data_dir,
            sft_run_name=f"{cfg.run_name}_sft_r{i}",
            model_dir=curr_model,
            train_from_scratch=train_from_scratch,
        )
        logger.info(f"Trained SFT model: {sft_dir}")
        curr_model = sft_dir

        # Take best checkpoint of trained model and get calibrated best likelihood range

        temps = get_temperatures(cfg, file_client, sft_dir, prev_hd)
        iter_gen_outputs, hd = run_iterative_generation(
            cfg,
            file_client,
            combined_sft_dataset_fp,
            ga_data_dir,
            sft_dir,
            higher_score_particle_field="higher_score_particle",
            lower_score_particle_field="lower_score_particle",
            higher_score_field="higher_score",
            lower_score_field="lower_score",
            temps=temps,
        )
        prev_hd = hd
        logger.info(f"Iterative generations output path: {iter_gen_outputs}")
        prev_round_outputs_fp = iter_gen_outputs

    all_prev_pref_datasets = []
    prev_hd = None
    for i in tqdm(range(cfg.num_dpo_rounds), desc="DPO iterations"):
        n = cfg.num_labels_after_first_round
        dpo_dataset_fp = create_propen_preference_dataset(
            cfg, file_client, prev_round_outputs_fp, filename_prefix=f"dpo_r{i}_", n=n
        )
        combined_dpo_dataset_fp = combine_new_with_old_datasets(
            cfg, file_client, all_prev_pref_datasets, dpo_dataset_fp
        )
        logger.info(f"DPO training dataset: {combined_dpo_dataset_fp}")
        all_prev_pref_datasets.append(dpo_dataset_fp)

        # Consider picking last checkpoint that generated >90% parsable particles w/ correct length and vals in range
        dpo_dir = train_dpo(
            cfg,
            file_client,
            curr_model,
            combined_dpo_dataset_fp,
            ga_data_dir,
            run_name=f"{cfg.run_name}_dpo_r{i}",
        )
        logger.info(f"DPO model trained in {dpo_dir}.")
        curr_model = dpo_dir

        temps = get_temperatures(cfg, file_client, dpo_dir, prev_hd)
        iter_gen_outputs, hd = run_iterative_generation(
            cfg,
            file_client,
            combined_dpo_dataset_fp,
            ga_data_dir,
            dpo_dir,
            higher_score_particle_field="prompt",
            lower_score_particle_field="chosen",
            higher_score_field="prompt_score",
            lower_score_field="chosen_score",
            temps=temps,
        )
        prev_hd = hd
        logger.info(
            f"Iterative generations output path (from DPO model): {iter_gen_outputs}"
        )
        prev_round_outputs_fp = iter_gen_outputs

    prev_hd = None
    for i in tqdm(range(cfg.num_marge_rounds), desc="MargE iterations"):
        n = cfg.num_labels_after_first_round
        marge_dataset_fp = create_propen_sft_dataset(
            cfg,
            file_client,
            prev_round_outputs_fp,
            filename_prefix=f"marge_r{i}_",
            n=n,
            allow_same_score_pair=False,
        )
        combined_marge_dataset_fp = combine_new_with_old_datasets(
            cfg, file_client, all_prev_sft_datasets, marge_dataset_fp
        )
        logger.info(f"MargE training dataset: {combined_marge_dataset_fp}")
        all_prev_sft_datasets.append(marge_dataset_fp)

        marge_dir = train_marge(
            cfg,
            file_client,
            curr_model,
            combined_marge_dataset_fp,
            ga_data_dir,
            run_name=f"{cfg.run_name}_marge_r{i}",
        )
        logger.info(f"MargE model trained in {marge_dir}.")
        curr_model = marge_dir

        temps = get_temperatures(cfg, file_client, marge_dir, prev_hd)
        iter_gen_outputs, hd = run_iterative_generation(
            cfg,
            file_client,
            combined_marge_dataset_fp,
            ga_data_dir,
            marge_dir,
            higher_score_particle_field="higher_score_particle",
            lower_score_particle_field="lower_score_particle",
            higher_score_field="higher_score",
            lower_score_field="lower_score",
            temps=temps,
        )
        prev_hd = hd
        logger.info(
            f"Iterative generations output path (from MargE model): {iter_gen_outputs}\nAverage hamming distance: {hd}"
        )
        prev_round_outputs_fp = iter_gen_outputs


if __name__ == "__main__":
    main()
