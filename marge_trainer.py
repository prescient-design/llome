import inspect
import json
import os
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer import TRAINER_STATE_NAME
from transformers.utils import logging as transformers_logging
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, PREFIX_CHECKPOINT_DIR

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import (
    RunningMoments,
    SyncRefModelCallback,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

logger = transformers_logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


@dataclass
class MargeDataCollatorWithPadding:
    r"""
    MargE DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False
    input_field_name: str = "input"
    target_field_name: str = "target"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith(self.input_field_name)) and (
                        k.endswith("input_ids")
                    ):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(
                        (self.input_field_name, self.target_field_name)
                    ) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if self.input_field_name in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                    # for the prompt, flip back so padding is on left side
                    if self.input_field_name in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


@dataclass
class MargeConfig(DPOConfig):
    alpha: float = 1.0
    input_field_name: str = "input"
    target_field_name: str = "target"
    # score fields should NOT be prefixed by input_field_name or target_field_name
    input_score_field_name: str = "score_input"
    target_score_field_name: str = "score_target"
    self_normalize_weights: bool = False
    reinforce_style: bool = False


class MargeTrainer(Trainer):
    def __init__(
        self,
        metrics_fn: Callable[
            Tuple[torch.utils.data.Dataset, List[str]], Dict[str, Any]
        ] = None,
        rewards_fn: Callable[
            Dict[str, Union[List, torch.LongTensor]], torch.Tensor
        ] = None,
        num_generate_batches: int = 1,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        args: Optional[MargeConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        pretokenized: bool = False,  # Whether the datasets are already pre-tokenized.
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
        threshold_percent_valid: float = 0.9,  # for selecting the best checkpoint -- lower threshold of
        # percent particles generated by the checkpoint that must be valid
    ):
        self.metrics_fn = metrics_fn
        self.rewards_fn = rewards_fn
        self.num_generate_batches = num_generate_batches
        self.threshold_percent_valid = threshold_percent_valid
        self.reward_sum = 0.0
        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the MargeTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )
        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_init_kwargs` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the MargeTrainer/DPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["torch_dtype"] = (
                ref_model_init_kwargs["torch_dtype"]
                if ref_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, ref_model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the MargeTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the MargeTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model, **ref_model_init_kwargs
            )

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if force_use_ref_model:
            warnings.warn(
                "You passed `force_use_ref_model` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.force_use_ref_model = force_use_ref_model

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in MargeTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(
                model, "is_loaded_in_4bit", False
            ):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {
                    "use_gradient_checkpointing": args.gradient_checkpointing
                }

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = (
                        args.gradient_checkpointing_kwargs
                    )

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(
                        make_inputs_require_grad
                    )

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        if generate_during_eval:
            warnings.warn(
                "You passed `generate_during_eval` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.generate_during_eval = generate_during_eval
        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the MargeTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if reference_free:
            warnings.warn(
                "You passed `reference_free` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.reference_free = reference_free
        self.reference_free = args.reference_free

        if precompute_ref_log_probs:
            warnings.warn(
                "You passed `precompute_ref_log_probs` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.precompute_ref_log_probs = precompute_ref_log_probs

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is not None:
            warnings.warn(
                "You passed `max_length` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_length = max_length
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if max_prompt_length is not None:
            warnings.warn(
                "You passed `max_prompt_length` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_prompt_length = max_prompt_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if max_target_length is not None:
            warnings.warn(
                "You passed `max_target_length` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_target_length = max_target_length
        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_target_length = 128

        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        if data_collator is None:
            data_collator = MargeDataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
                input_field_name=args.input_field_name,
                target_field_name=args.target_field_name,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using MargeDataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = (
            args.padding_value if padding_value is not None else tokenizer.pad_token_id
        )
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.max_target_length = args.max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if label_smoothing != 0:
            warnings.warn(
                "You passed `label_smoothing` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_smoothing = label_smoothing

        if beta != 0.1:
            warnings.warn(
                "You passed `beta` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.beta = beta
        self.beta = args.beta
        self.alpha = args.alpha
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the MargeTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc
        self.args = args

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            self.pretokenized = pretokenized
            if not self.pretokenized:
                train_dataset = train_dataset.map(
                    self.tokenize_row, num_proc=self.dataset_num_proc
                )
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        self.tokenize_row, num_proc=self.dataset_num_proc
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if (
                self.accelerator.state.deepspeed_plugin.zero_stage == 3
                and self.precompute_ref_log_probs
            ):
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            if precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(self.train_dataset, **dataloader_params)
            )

            reference_target_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Train dataset reference log probs"
            ):
                reference_target_logp = self.compute_reference_log_probs(padded_batch)
                reference_target_logp = self.accelerator.gather_for_metrics(
                    reference_target_logp
                )
                reference_target_logps.append(reference_target_logp.cpu())

            all_reference_target_logps = (
                torch.cat(reference_target_logps).float().numpy()
            )

            self.train_dataset = self.train_dataset.add_column(
                name=f"reference_target_logps", column=all_reference_target_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(eval_dataset, **dataloader_params)
            )

            reference_target_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Eval dataset reference log probs"
            ):
                reference_target_logp = self.compute_reference_log_probs(padded_batch)
                reference_target_logp = self.accelerator.gather_for_metrics(
                    reference_target_logp
                )
                reference_target_logps.append(reference_target_logp.cpu())

            all_reference_target_logps = (
                torch.cat(reference_target_logps).float().numpy()
            )

            eval_dataset = eval_dataset.add_column(
                name="reference_target_logps", column=all_reference_target_logps
            )

            # Save calculated reference_target_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][
            len(prompt_input_ids) :
        ]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError(
                "Prompt input ids and answer input ids should have the same length."
            )

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if (
            prompt_input_ids
            != full_tokenized["input_ids"][:response_token_ids_start_idx]
        ):
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][
            :response_token_ids_start_idx
        ]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError(
                "Prompt input ids and attention mask should have the same length."
            )

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][
            response_token_ids_start_idx:
        ]

        return {
            f"{self.args.input_field_name}_input_ids": prompt_input_ids,
            f"{self.args.input_field_name}_attention_mask": prompt_attention_mask,
            "input_ids": answer_input_ids,
            "attention_mask": answer_attention_mask,
        }

    def tokenize_row(
        self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None
    ) -> Dict:
        """Tokenize a single row from a MargE specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the target responses, which are of length equal to
            the sum of the length of the prompt and the target response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature[self.args.input_field_name]
        target = feature[self.args.target_field_name]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {
                f"{self.args.input_field_name}_{k}": v for k, v in prompt_tokens.items()
            }

            if not isinstance(target, str):
                raise ValueError(f"target should be an str but got {type(target)}")
            target_tokens = self.build_tokenized_answer(prompt, target)

            target_prompt_len_input_ids = len(
                target_tokens[f"{self.args.input_field_name}_input_ids"]
            )
            prompt_len_input_ids = target_prompt_len_input_ids

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            longer_response_length = len(target_tokens["input_ids"])

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [target_tokens, prompt_tokens]:
                if (
                    len(answer_tokens[f"{self.args.input_field_name}_input_ids"])
                    + longer_response_length
                    > self.max_length
                ):
                    if self.truncation_mode == "keep_start":
                        for k in [
                            f"{self.args.input_field_name}_input_ids",
                            f"{self.args.input_field_name}_attention_mask",
                        ]:
                            answer_tokens[k] = answer_tokens[k][
                                : self.max_prompt_length
                            ]
                    elif self.truncation_mode == "keep_end":
                        for k in [
                            f"{self.args.input_field_name}_input_ids",
                            f"{self.args.input_field_name}_attention_mask",
                        ]:
                            answer_tokens[k] = answer_tokens[k][
                                -self.max_prompt_length :
                            ]
                    else:
                        raise ValueError(
                            f"Unknown truncation mode: {self.truncation_mode}"
                        )

            # if that's still too long, truncate the response
            for answer_tokens in [target_tokens]:
                if (
                    len(answer_tokens[f"{self.args.input_field_name}_input_ids"])
                    + longer_response_length
                    > self.max_length
                ):
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][
                            : self.max_length - self.max_prompt_length
                        ]

            # Create labels
            target_sequence_tokens = {
                k: target_tokens[f"{self.args.input_field_name}_{k}"] + target_tokens[k]
                for k in ["input_ids", "attention_mask"]
            }
            target_sequence_tokens["labels"] = target_sequence_tokens["input_ids"][:]
            target_sequence_tokens["labels"][
                : len(target_tokens[f"{self.args.input_field_name}_input_ids"])
            ] = [self.label_pad_token_id] * len(
                target_tokens[f"{self.args.input_field_name}_input_ids"]
            )

            for k, toks in {
                f"{self.args.target_field_name}_": target_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            target_tokens = self.tokenizer(
                target,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )

            batch[f"{self.args.target_field_name}_labels"] = target_tokens["input_ids"]
            batch[f"{self.args.input_field_name}_input_ids"] = prompt_tokens[
                "input_ids"
            ]
            batch[f"{self.args.input_field_name}_attention_mask"] = prompt_tokens[
                "attention_mask"
            ]

            if model is not None and hasattr(
                model, "prepare_decoder_input_ids_from_labels"
            ):
                batch[f"{self.args.target_field_name}_decoder_input_ids"] = (
                    model.prepare_decoder_input_ids_from_labels(
                        labels=torch.tensor(
                            batch[f"{self.args.target_field_name}_labels"]
                        )
                    )
                )

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_target_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_target_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_target_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
        target_field_name: str = "target",
        input_field_name: str = "prompt",
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys '{self.args.target_field_name}_input_ids', which contains tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.
            target_field_name: The name of the target field.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = batch[f"{target_field_name}_labels"].shape[1]
        else:
            max_length = batch[f"{target_field_name}_input_ids"].shape[1]

        for k in batch:
            if k.startswith(target_field_name) and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace(target_field_name, "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = (
                batch[f"{input_field_name}_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch[f"{input_field_name}_attention_mask"]
                .repeat(2, 1)
                .to(device=device)
            )
        return concatenated_batch

    def marge_loss(
        self,
        policy_target_logps: torch.FloatTensor,
        policy_reference_target_prob_ratios: torch.FloatTensor,
        target_rewards: torch.FloatTensor,
        target_sizes: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the MargE loss for a batch of policy and reference model log probabilities.

        Args:
            policy_target_logps: Log probabilities of the policy model for the targets. Shape: (batch_size,)
            policy_reference_target_prob_ratios: Ratios of pi_theta(y) / pi_ref(y). Shape: (batch_size,)
            target_rewards: Rewards for the targets. Shape: (batch_size,)
            target_sizes: Number of tokens in each y. Shape: (batch_size,)

        Returns:
            losses: Tensor containing loss value for each example in the batch. Shape: (batch_size,)
            kl_div_to_target
            kl_div_to_ref
        """
        target_sizes = target_sizes.to(policy_target_logps.device)
        kl_div_to_ref = -policy_target_logps / target_sizes

        if self.args.self_normalize_weights:
            policy_reference_target_prob_ratios = (
                policy_reference_target_prob_ratios
                / policy_reference_target_prob_ratios.sum()
            )
        kl_div_to_target = policy_reference_target_prob_ratios * (
            policy_target_logps / target_sizes
            - (target_rewards.to(policy_target_logps.device))
        )
        if self.args.reinforce_style:
            avg_reward_baseline = self.reward_sum / (self.state.global_step + 1)
            losses = (
                -policy_target_logps * ((target_rewards).to(policy_target_logps.device))
                + self.beta * kl_div_to_ref
            )
            self.reward_sum += target_rewards.mean().detach().cpu().item()
            return losses, target_rewards, avg_reward_baseline
        else:
            losses = self.alpha * kl_div_to_target + self.beta * kl_div_to_ref
        return losses, kl_div_to_target, kl_div_to_ref

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs."""
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            target_field_name=self.args.target_field_name,
            input_field_name=self.args.input_field_name,
        )

        labels = concatenated_batch["concatenated_labels"]
        model_kwargs = (
            {
                "labels": labels,
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        return (all_logps, all_logits, size_completion)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (policy_target_logps, policy_target_logits, target_num_tokens) = (
            self.concatenated_forward(model, batch)
        )

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_target_logps" in batch:
            reference_target_logps = batch["reference_target_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_target_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_target_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        policy_reference_target_ratios = torch.exp(
            policy_target_logps - reference_target_logps
        )
        rewards = self.rewards_fn(batch)
        prefix = "eval_" if train_eval == "eval" else ""
        if self.args.reinforce_style:
            losses, target_rewards, avg_reward_baseline = self.marge_loss(
                policy_target_logps,
                policy_reference_target_ratios,
                rewards,
                target_num_tokens,
            )
            metrics[f"{prefix}rewards/target"] = target_rewards.mean().cpu()
            metrics[f"{prefix}rewards/running_average"] = avg_reward_baseline
        else:
            losses, kl_div_to_target, kl_div_to_ref = self.marge_loss(
                policy_target_logps,
                policy_reference_target_ratios,
                rewards,
                target_num_tokens,
            )
            metrics[f"{prefix}kl_div/reverse_to_target"] = kl_div_to_target.mean().cpu()
            metrics[f"{prefix}kl_div/forward_to_ref"] = kl_div_to_ref.mean().cpu()

        metrics[f"{prefix}likelihoods/policy_reference_ratio"] = (
            policy_reference_target_ratios.mean().cpu()
        )

        metrics[f"{prefix}logps/policy"] = policy_target_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/target"] = policy_target_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for MargeDataCollatorWithPadding, and you passed a datacollator that is different than "
                "MargeDataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="train"
            )

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(
        self, model, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
        """
        Greedily decode samples from the model and reference model for the given batch of inputs.
        """

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            nullcontext
            if not self._peft_has_been_casted_to_bf16
            else torch.cuda.amp.autocast
        )

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch[f"{self.args.input_field_name}_input_ids"],
                attention_mask=batch[f"{self.args.input_field_name}_attention_mask"],
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=None,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch[f"{self.args.input_field_name}_input_ids"],
                            attention_mask=batch[
                                f"{self.args.input_field_name}_attention_mask"
                            ],
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            max_length=None,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch[f"{self.args.input_field_name}_input_ids"],
                        attention_mask=batch[
                            f"{self.args.input_field_name}_attention_mask"
                        ],
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        max_length=None,
                    )

        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output[
                :, batch[f"{self.args.input_field_name}_input_ids"].shape[-1] :
            ],
            skip_special_tokens=True,
        )

        reference_output_decoded = self.tokenizer.batch_decode(
            reference_output[
                :, batch[f"{self.args.input_field_name}_input_ids"].shape[-1] :
            ],
            skip_special_tokens=True,
        )

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for MargeDataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="eval"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/target": metrics["eval_logits/target"],
        }
        logits = tuple(
            v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys
        )
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # self.get_eval_dataloader(dataset) # TODO: also eval on best examples from overall dataset?

        # Sample and save to game log if requested (for |self.num_generate_batches| batches to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            if self.num_generate_batches is not None:
                num_samples = len(dataloader.dataset)
                random.seed(self.args.seed)
                random_indices = random.sample(
                    range(num_samples),
                    k=min(
                        num_samples,
                        self.args.eval_batch_size * self.num_generate_batches,
                    ),
                )

                # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
                random_batches_dataset = dataloader.dataset.select(random_indices)
            else:
                random_indices = list(
                    np.arange(0, len(dataloader.dataset), self.args.eval_batch_size)
                )
                random_batches_dataset = dataloader.dataset
            input_particles = [
                d[self.args.input_field_name] for d in random_batches_dataset
            ]
            input_scores = [
                d[self.args.input_score_field_name] for d in random_batches_dataset
            ]
            policy_output_decoded = []
            ref_output_decoded = []
            for i in range(0, len(random_indices), self.args.eval_batch_size):
                random_batch = random_batches_dataset.select(
                    range(
                        i,
                        min(i + self.args.eval_batch_size, len(random_batches_dataset)),
                    )
                )
                random_batch = self._prepare_inputs(self.data_collator(random_batch))
                policy_output_decoded_batch, ref_output_decoded_batch = (
                    self.get_batch_samples(self.model, random_batch)
                )
                policy_output_decoded.extend(policy_output_decoded_batch)
                ref_output_decoded.extend(ref_output_decoded_batch)

            policy_metrics = self.metrics_fn(
                random_batches_dataset,
                policy_output_decoded,
            )
            policy_metrics = {f"eval_policy_{k}": v for k, v in policy_metrics.items()}
            self.store_metrics(policy_metrics, train_eval="eval")
            ref_metrics = self.metrics_fn(
                random_batches_dataset,
                ref_output_decoded,
            )
            ref_metrics = {f"eval_ref_{k}": v for k, v in ref_metrics.items()}
            self.store_metrics(ref_metrics, train_eval="eval")

        # Base evaluation
        initial_output = super(MargeTrainer, self).evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        initial_output = EvalLoopOutput(
            predictions=initial_output.predictions,
            label_ids=initial_output.label_ids,
            metrics={**initial_output.metrics, **self._stored_metrics["eval"]},
            num_samples=initial_output.num_samples,
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(
            model=self.model, tag_names=self._tag_names, kwargs=kwargs
        )

        return super().push_to_hub(
            commit_message=commit_message, blocking=blocking, **kwargs
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                # also check that the checkpoint produces mostly valid outputs
                validity_higher_metrics = [
                    "eval_policy_%_parsable",
                    "eval_policy_%_correct_length",
                    "eval_policy_%_feasible",
                    "eval_policy_%_values_in_range",
                ]
                validity_lower_metrics = [
                    "eval_policy_%_repeated_input",
                ]
                passed_validity_check = True
                for m in validity_higher_metrics:
                    if m not in metrics:
                        logger.warning(f"Metric {m} not found in metrics: {metrics}")
                        continue
                    elif metrics[m] < self.threshold_percent_valid:
                        logger.warning(
                            f"Checkpoint {self.state.global_step} has best {metric_to_check} "
                            + f"but {m} is too low: {metrics[m]} < {self.threshold_percent_valid}"
                        )
                        passed_validity_check = False
                        break
                for m in validity_lower_metrics:
                    if m not in metrics:
                        logger.warning(f"Metric {m} not found in metrics: {metrics}")
                        continue
                    elif metrics[m] > 1.0 - self.threshold_percent_valid:
                        logger.warning(
                            f"Checkpoint {self.state.global_step} has best {metric_to_check} "
                            + f"but {m} is too high: {metrics[m]} > {1.0 - self.threshold_percent_valid}"
                        )
                        passed_validity_check = False
                        break
                if passed_validity_check:
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            # Update the `TrainerControl` state to where we are currently
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
