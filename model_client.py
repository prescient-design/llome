import gc
import logging
import numpy as np
import os
import s3fs
import tempfile
import torch
import torch.distributed.checkpoint as dist_cp

from finetune_utils import maybe_log
from string import Template
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple, Union


class ModelClient:
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        hf_model_name: str = None,  # The name of the model in the HF hub. Only necessary
        # if loading from a sharded FSDP checkpoint.
        logger: logging.Logger = None,
        max_generate_length: int = 500,
        device: Optional[str] = None,
        **model_init_args,
    ):
        self.model_name_or_path = model_name_or_path
        self.hf_model_name = hf_model_name
        self.logger = logger

        self.max_generate_length = max_generate_length

        self.device = device
        self.model_init_args = model_init_args

        if model_name_or_path.startswith("s3://"):
            self._load_from_s3(model_name_or_path)
        elif (
            os.path.exists(model_name_or_path)
            and not os.path.exists(f"{model_name_or_path}/config.json")
            and os.path.exists(f"{model_name_or_path}/pytorch_model_fsdp_0")
        ):
            maybe_log(
                self.logger,
                f"{model_name_or_path} is a sharded FSDP model. Attempting to consolidate and convert.",
                level="info",
            )
            self.model, self.config, self.tokenizer = self._convert_checkpoint(
                hf_model_name=hf_model_name,
                fsdp_model_path=f"{model_name_or_path}/pytorch_model_fsdp_0",
                output_path=model_name_or_path,
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            extra_model_kwargs = {**self.model_init_args}
            if "gg-hf/gemma" in self.model_name_or_path:
                extra_model_kwargs["torch_dtype"] = torch.bfloat16
                extra_model_kwargs["device_map"] = "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True, **extra_model_kwargs
            )
            if self.device is not None:
                self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError(
                    f"Tokenizer for {self.model_name_or_path} does not have a PAD token."
                )
        if hasattr(self.config, "max_position_embeddings"):
            self.max_length = self.config.max_position_embeddings
        elif hasattr(self.config, "max_sequence_length"):
            self.max_length = self.config.max_sequence_length
        elif hasattr(self.config, "n_positions"):
            self.max_length = self.config.n_positions
        else:
            raise ValueError(
                f"Could not find max_position_embeddings, n_positions, or max_sequence_length in model config for {self.model_name_or_path}"
            )

    def _load_from_s3(self, s3_uri: str):
        self.s3 = s3fs.S3FileSystem()
        with tempfile.TemporaryDirectory() as td:
            maybe_log(self.logger, f"Downloading model from {s3_uri} into {td}.")
            if not td.endswith("/"):
                td += "/"
            if not s3_uri.endswith("/"):
                s3_uri += "/"
            s3_uri += (
                "*"  # will copy only the files in the directory, not any nested ones
            )
            self.s3.get(s3_uri, td)
            self.config = AutoConfig.from_pretrained(td)
            self.model_init_args["trust_remote_code"] = True
            self.model = AutoModelForCausalLM.from_pretrained(
                td, **self.model_init_args
            )
            if self.device is not None:
                self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(td)
        gc.collect()
        torch.cuda.empty_cache()

    def _convert_checkpoint(
        self, hf_model_name: str, fsdp_model_path: str, output_path: str
    ):
        """
        hf_model_name: Name of model in HF Hub, e.g. "gpt2".
        fsdp_model_path: path to the fsdp checkpoint, for example `/x/checkpoint-xxx/pytorch_model_fsdp_x`
        output_path: output path to save the converted checkpoint
        """
        config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, **self.model_init_args
        ).cuda()
        model = self._load_sharded_model_single_gpu(model, fsdp_model_path)
        model.save_pretrained(output_path, max_shard_size="10GB")
        tokenizer.save_pretrained(output_path)
        return model, config, tokenizer

    def _load_sharded_model_single_gpu(self, model, model_path):
        state_dict = {"model": model.state_dict()}

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(model_path),
            no_dist=True,
        )

        result = model.load_state_dict(state_dict["model"])
        maybe_log(
            self.logger,
            f"Sharded state checkpoint loaded from {model_path}. Result: {result}",
            level="info",
        )
        return model

    def _apply_chat_template(self, msgs) -> str:
        if self.tokenizer.chat_template is not None:
            chat_msgs = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            raise NotImplementedError(
                f"Chat template does not exist for model {self.model_name_or_path}."
            )
        return chat_msgs

    def _tokenize_batch(
        self,
        text_batch: Union[List[List[Dict[str, str]]], List[str]],
        max_generate_length: Optional[int] = None,
    ) -> torch.tensor:
        """Tokenize either a batch of messages or a single text string."""
        if not text_batch:
            return None
        if max_generate_length is not None:
            max_gen_length = max_generate_length
        else:
            max_gen_length = self.max_generate_length
        max_context_length = self.max_length - max_gen_length
        if isinstance(text_batch[0], list):
            # text batch is list of messages, so apply chat template
            texts_to_tokenize = [self._apply_chat_template(text) for text in text_batch]
        else:
            texts_to_tokenize = text_batch
        tokenized = self.tokenizer(
            texts_to_tokenize,
            truncation=True,
            padding="longest",
            max_length=max_context_length,
            return_tensors="pt",
        )
        return tokenized

    def _chat_hf_model(
        self, msgs: Union[List[Dict[str, str]], str], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion for either a series of messages or a single text prompt.
        """
        tokenized = self._tokenize_batch([msgs]).input_ids
        if self.device is not None:
            tokenized = tokenized.to(self.device)
        outputs = self.model.generate(tokenized, **kwargs)
        return {"output_strs": self.tokenizer.batch_decode(outputs)}

    def chat_single_turn(self, msgs: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return self._chat_hf_model(msgs, **kwargs)

    def chat_raw_logits(
        self, input_ids: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        """Alternative API for getting all logits."""
        return self.model(input_ids, attention_mask=attention_mask).logits

    def chat_single_turn_text(self, text: str, **kwargs) -> Dict[str, Any]:
        msgs = [{"role": "user", "content": text}]
        return self._chat_hf_model(msgs, **kwargs)

    @torch.no_grad()
    def generate_texts_batched(
        self,
        input_texts: List[str],
        batch_size: int = 8,
        return_likelihoods=False,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]]:
        """
        Runs batched generation. If return_likelihoods=True, returns a tuple containing
        the output sequences, the output tokens, and the output token log-likelihoods.
        Otherwise, returns only the output sequences.
        """
        output_strs = []
        all_output_token_ids = []
        all_token_logps = []
        for batch_start_idx in tqdm(
            range(0, len(input_texts), batch_size), desc="Batched generation..."
        ):
            batch_end_idx = batch_start_idx + batch_size
            batch_texts = input_texts[batch_start_idx:batch_end_idx]
            input_ids = self._tokenize_batch(batch_texts).input_ids
            if self.device is not None:
                input_ids = input_ids.to(self.device)
            gen_args = {**kwargs}
            if return_likelihoods:
                gen_args["return_dict_in_generate"] = True
                gen_args["output_scores"] = True
            outputs = self.model.generate(
                input_ids, **gen_args, tokenizer=self.tokenizer
            )
            output_token_ids = outputs.sequences if return_likelihoods else outputs
            # remove the input from each output
            decoded_outputs = self.tokenizer.batch_decode(
                output_token_ids[:, input_ids.shape[-1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            output_strs.extend(decoded_outputs)
            if return_likelihoods:
                token_logps = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                for oti in output_token_ids[:, input_ids.shape[-1] :]:
                    all_output_token_ids.append(oti)
                for o_logps in token_logps:
                    all_token_logps.append(o_logps)
        if return_likelihoods:
            return (output_strs, all_output_token_ids, all_token_logps)
        return output_strs

    @torch.no_grad()
    def compute_likelihoods(
        self,
        inputs: List[str],
        targets: List[str],
        batch_size: int = 16,
        device: str = "cuda",
        # add_start_token: bool = True,
        logger: logging.Logger = None,
    ) -> List[float]:
        # Compute length-normalized likelihoods of target tokens given the input
        # TODO: write test to check that the logprobs match what model.generate and compute_transition_scores gives!
        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be in ['gpu', 'cpu', 'cuda']."
            if device == "gpu":
                device = "cuda"
        else:
            device = self.device

        all_likelihoods = []
        for start_index in tqdm(
            range(0, len(inputs), batch_size), desc="Computing log likelihood..."
        ):
            end_index = min(start_index + batch_size, len(inputs))
            batch = [
                f"{input}{target}"
                for input, target in zip(
                    inputs[start_index:end_index], targets[start_index:end_index]
                )
            ]
            tokenized = self._tokenize_batch(batch, max_generate_length=0).to(
                self.device
            )
            outputs = self.model(**tokenized)
            scores = outputs.logits
            probs = torch.exp(torch.nn.functional.log_softmax(scores, dim=-1))
            # get probs and mask out both padding and input tokens
            inputs_tokenized = self._tokenize_batch(inputs[start_index:end_index])
            input_seq_lens = inputs_tokenized.attention_mask.sum(-1)
            input_tokens_mask = torch.LongTensor(
                [
                    [0 for _ in range(input_seq_lens[i].item())]
                    + [
                        1
                        for _ in range(
                            tokenized.input_ids.shape[1] - input_seq_lens[i].item()
                        )
                    ]
                    for i in range(tokenized.input_ids.shape[0])
                ]
            ).to(self.device)
            indexes = tokenized.input_ids[:, 1:]
            next_token_probs = (
                torch.gather(probs, -1, indexes.unsqueeze(-1)).squeeze(-1)
                * tokenized.attention_mask[:, 1:]
                * input_tokens_mask[:, 1:]
            )
            next_token_probs = next_token_probs.cpu().numpy()
            targets_tokenized = self._tokenize_batch(targets[start_index:end_index])
            targets_seq_lens = targets_tokenized.attention_mask.sum(-1).cpu().numpy()
            avg_likelihoods = list(next_token_probs.sum(-1) / targets_seq_lens)
            all_likelihoods.extend(avg_likelihoods)
        return all_likelihoods
