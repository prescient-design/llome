import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid

from typing import List, Tuple


@contextlib.contextmanager
def rm_slurm_env():
    """Context manager for temporarily removing slurm environment variables that
    may be mistakenly inherited by new slurm jobs launched from the current program.
    """
    old_environ = {
        x: os.environ.pop(x, None)
        for x in [
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_CPU_BIND",
            "SLURM_CPU_BIND_TYPE",
            "SLURM_CPU_BIND_LIST",
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_LOCALID",
            "SLURM_NODEID",
            "SLURM_NTASKS",
            "SLURM_PROCID",
        ]
    }
    try:
        yield
    finally:
        for k, val in old_environ.items():
            if val is not None:
                os.environ[k] = val


def submit_sbatch_job(
    sbatch_cmd: str, dump_dir: str, blocking: bool = True
) -> Tuple[subprocess.Popen, str]:
    """Submit sbatch job, get slurm job ID, and decide whether to wait or not.
    Returns both the process and the slurm job ID
    """
    sbatch_output = os.path.join(dump_dir, f"sbatch_output_{uuid.uuid1()}")
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(sbatch_cmd)
        f.close()
        p = subprocess.Popen(
            args=[f"sbatch --wait --parsable {f.name} > {sbatch_output}"],
            shell=True,
            executable="/bin/bash",
        )
        # now wait for the sbatch_output file to appear
        logging.info(f"Waiting for job submission...")
        while not os.path.exists(sbatch_output) or os.path.getsize(sbatch_output) == 0:
            time.sleep(1)
        # get job ID
        slurm_job_id = open(sbatch_output).read().strip()
        logging.info(f"Submitted slurm job {slurm_job_id}")
        os.remove(sbatch_output)
        # copy submission file to current directory with job ID in filename
        new_submission_fp = os.path.join(dump_dir, f"{slurm_job_id}_submission.sbatch")
        shutil.copy(f.name, new_submission_fp)
        logging.info(
            f"Slurm submission script for job {slurm_job_id} written to {new_submission_fp}."
        )
        if blocking:
            logging.info(f"Waiting for job {slurm_job_id}...")
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Slurm job {slurm_job_id} failed.")
            logging.info(f"Slurm job {slurm_job_id} succeeded")
        return p, slurm_job_id


@rm_slurm_env()
def submit_cmd_to_slurm(
    py_cmd: str,
    conda_env: str,
    dump_dir: str,
    blocking: bool = False,
    setup_str: str = "",  # setup  that goes between sbatch args and running the py_cmd
    **slurm_kwargs,
) -> Tuple[subprocess.Popen, str]:
    sbatch_str = f"source ~/.bashrc\ncd ~/llome\nsource .venv/bin/activate\n{py_cmd}"
    # add #SBATCH commands on top
    if "output" not in slurm_kwargs:
        slurm_kwargs["output"] = f"{dump_dir}/%x_%j.logs"
    if "error" not in slurm_kwargs:
        slurm_kwargs["error"] = f"{dump_dir}/%x_%j.logs"
    sbatch_prefix = ["#!/usr/bin/env bash"]
    sbatch_prefix.extend(
        [f"#SBATCH --{k.replace('_', '-')}={v}" for k, v in slurm_kwargs.items()]
    )
    sbatch_str = "\n".join(sbatch_prefix) + "\n\n" + setup_str + "\n\n" + sbatch_str
    return submit_sbatch_job(sbatch_str, dump_dir, blocking=blocking)


def wait_for_slurm_jobs_to_complete(jobs: List[Tuple[subprocess.Popen, str]]):
    """Given a list of tuples of (process, slurm job ID), wait for all to complete."""
    processes = ", ".join([f"{p.pid}" for (p, _) in jobs])
    job_ids = ", ".join([f"{job_id}" for (_, job_id) in jobs])
    logging.info(
        f"Waiting for processes {processes} (slurm jobs {job_ids}) to complete..."
    )
    for p, job_id in jobs:
        rc = p.wait()
        stdout, stderr = p.communicate()
        logging.info(f"slurm job {job_id} stdout: {stdout}")
        logging.info(f"slurm job {job_id} stderr: {stderr}")
        if rc != 0:
            raise RuntimeError(f"Process {p.pid} (slurm job {job_id}) failed!")
        logging.info(f"Slurm job {job_id} succeeded!")
