# LLOME

This is the repository for the paper ["Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks,"](https://arxiv.org/abs/2410.22296) by Angelica Chen<sup>\*</sup>, Samuel D. Stanton<sup>\*</sup>, Frances Ding, Robert G. Alberstein, Andrew M. Watkins, Richard Bonneau, Vladimir Gligorijević, Kyunghyun Cho, Nathan C. Frey

## Set-Up
1. Clone this repo to your home directory! (Some of the scripts in this repo assume that your code is stored in `~/sherpa`.)
1. Install the requirements in a virtual environment: 
```
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```
1. Run `aws configure` to ensure you are configured for access to S3 (if using S3 storage).

## To run a sanity check to ensure that your set-up is working
```
python -m run_pipeline --config-name=pipeline_sanity_check_marge local_output_dir=<PATH_TO_LOCAL_OUTPUT_DIR> parent_output_dir=<S3_OUTPUT_DIR>
```

## Important things to know about the LLOME pipeline
- **Storage**: This pipeline makes use of both local storage (`local_output_dir` in the YAML config) and S3 storage (`parent_output_dir` in the YAML config). All model checkpoints are automatically copied to S3 and deleted from local storage after the end of a training job. By default, all generated seeds, generations, and training data are stored in `<config.parent_output_dir>/<config.run_name>`, where `config` is the Hydra config. 
    - **To disable this behavior and use only local storage**, simply set `parent_output_dir: "null"` in the config.
- **Slurm Configurations**: All training, dataset generation/formatting, and generation jobs are launched as separate SLURM jobs. These SLURM jobs are easily configured with the respective `slurm_args` subsections in the config.
- **Resuming jobs**: This pipeline is designed to automatically resume previous runs, assuming that the pipeline is run with the same arguments as a prior run. To disable this behavior, use `--overwrite=True`.
- **Training large models**: We currently use DDP (without accompanying model parallelism) to train our models (FSDP is a TODO though!), which means that you likely need GPU RAM that is ~4x the size of the model to train in full precision. Our code is currently only tested on single-node multi-GPU setups.
- **Analyzing results**: To assess the performance of the LLM at iteration $i$, analyze the *training data* generated for iteration $i+1$ -- this accounts for both the likelihood-based ranking and filtering of the LLM at iteration $i$.

## To reproduce experiments
### For function $f_2$:

LLOME-SFT:
```python -m run_pipeline --config-name=pipeline_sft_f2.yaml local_output_dir=<PATH_TO_LOCAL_OUTPUT_DIR> parent_output_dir=<S3_OUTPUT_DIR>
```

LLOME-MargE:
```
python -m run_pipeline --config-name=pipeline_marge_f2.yaml local_output_dir=<PATH_TO_LOCAL_OUTPUT_DIR> parent_output_dir=<S3_OUTPUT_DIR>
```