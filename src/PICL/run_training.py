# -*- coding: utf-8 -*-
# Robust DPO training script: supports multiple data mixing strategies and PIC regularization

import multiprocessing
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import PartialState
from datasets import Dataset, interleave_datasets, load_dataset  
from peft import prepare_model_for_kbit_training
from robust_dpo import RobustDataCollator, RobustDPOConfig, RobustDPOTrainer

# Imports
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import get_peft_config, get_quantization_config


# Helper: print only from main process
def main_process_print(*args, **kwargs):
    """Print only from the main process (rank 0) to avoid duplicated logs in multi-GPU runs."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args, **kwargs)

# --- Script-specific arguments ---
@dataclass
class ScriptArguments:
    # Script arguments: define CLI options for the training runner

    data_files: str = field(
        default="data.jsonl",
        metadata={"help": "Path to training data file (JSONL format)"}
    )

    # Data mixing strategy
    mixing_strategy: str = field(
        default="original",
        metadata={"help": "Mixing strategy: 'original' or 'weighted'"}
    )

    mixing_ratios: str = field(
        default="0.4,0.3,0.3",
        metadata={"help": "Ratios when strategy='weighted' in order: SFT, Persona-DPO, Normal-DPO"}
    )

    stopping_strategy: str = field(
        default="all_exhausted",
        metadata={"help": "Stopping strategy for interleave_datasets: 'first_exhausted' or 'all_exhausted'"}
    )

def main():
    # Main: execute the training workflow
    
    # 1. Parsing command line arguments
    parser = HfArgumentParser([RobustDPOConfig, ScriptArguments, ModelConfig])  # type: ignore
    dpo_args, script_args, model_config = parser.parse_args_into_dataclasses()
    
    # Configure gradient checkpointing and DDP to avoid common issues
    if dpo_args.gradient_checkpointing:
        dpo_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    dpo_args.ddp_find_unused_parameters = False
    
    # Disable dataloader shuffle to preserve fixed batch ordering
    dpo_args.dataloader_shuffle = False

    # 2. Initialize distributed state (used for multi-GPU setups)
    distributed_state = PartialState()

    # Print high-level configuration without exposing sensitive values
    main_process_print("Model: [REDACTED]")
    try:
        lora_targets = model_config.lora_target_modules
        lora_summary = f"{len(lora_targets)} modules" if isinstance(lora_targets, (list, tuple)) else "[REDACTED]"
    except Exception:
        lora_summary = "[REDACTED]"
    main_process_print(f"LoRA Target Modules: {lora_summary}")
    main_process_print("Quantization: [REDACTED]")

    # 3. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Set pad_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Data processing function
    # Supports data types: Normal-DPO, Persona-DPO, Normal-SFT, Persona-SFT
    def process_and_tokenize(examples):
        # initialize return lists
        prompts = []
        full_prompts_input_ids = []
        persona_masks = []
        is_dpo_flags = []
        rejected_responses = []

        # store clean prompts (for DPO samples only)
        clean_prompts_input_ids = []

        types = examples.get("type", [None] * len(examples["prompt"]))
        rejecteds = examples.get("rejected", [None] * len(examples["prompt"]))

        for idx, (p, q, has_p) in enumerate(zip(
            examples["persona"], 
            examples["prompt"], 
            examples["has_persona"]
        )):
            # infer data type
            data_type = types[idx]
            if not data_type:
                r = rejecteds[idx]
                data_type = "sft" if (r is None or (isinstance(r, str) and len(r.strip()) == 0)) else "dpo"
            is_dpo = data_type != "sft"

            # build prompt (may include persona)
            user_content = (
                f"Answer the question according to the assigned persona.\n"
                f"persona: {p}\n"
                f"question: {q}"
            ) if (has_p and p) else q
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                main_process_print(f"Warning: Failed to apply chat template for sample {idx}: {e}. Using raw content.")
                formatted_prompt = user_content
            prompts.append(formatted_prompt)

            # build clean prompt ids only when PIC applies (DPO + persona)
            clean_input_id = []
            should_generate_clean = is_dpo and has_p and p
            
            if should_generate_clean:
                try:
                    formatted_clean_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    clean_tokens = tokenizer(formatted_clean_prompt, add_special_tokens=False)
                    clean_input_id = clean_tokens["input_ids"]
                    
                    # validate generated tokens are non-empty
                    if not clean_input_id or len(clean_input_id) == 0:
                        raise ValueError(f"Generated empty token list for clean prompt")
                except Exception as e:
                    main_process_print(f"Warning: Failed to create clean prompt for sample {idx} (prompt={q[:50]}...): {e}")
                    clean_input_id = []
            clean_prompts_input_ids.append(clean_input_id)

            # process main prompt input
            # compute offset_mapping only when persona detection is needed
            need_offsets = (has_p and is_dpo and p)
            tokenized_result = tokenizer(
                formatted_prompt,
                add_special_tokens=False,
                return_offsets_mapping=need_offsets
            )
            full_tokens = tokenized_result["input_ids"]
            
            # compute persona mask
            seq_len = len(full_tokens)
            mask = [0] * seq_len
            if need_offsets:
                offsets = tokenized_result["offset_mapping"]
                persona_start_char = formatted_prompt.find(p)
                if persona_start_char != -1:
                    persona_end_char = persona_start_char + len(p)
                    persona_start_token = None
                    persona_end_token = seq_len
                    for token_idx, (start_char, end_char) in enumerate(offsets):
                        if persona_start_token is None and end_char > persona_start_char:
                            persona_start_token = token_idx
                        if start_char >= persona_end_char:
                            persona_end_token = token_idx
                            break
                    if persona_start_token is not None:
                        mask[persona_start_token:persona_end_token] = [1] * (persona_end_token - persona_start_token)
                else:
                    p_tokens = tokenizer(p, add_special_tokens=False)["input_ids"]
                    conservative_len = min(seq_len, len(p_tokens) + 5)
                    mask[:conservative_len] = [1] * conservative_len
            
            full_prompts_input_ids.append(full_tokens)
            persona_masks.append(mask)
            is_dpo_flags.append(is_dpo)
            rejected_responses.append("" if not is_dpo else (rejecteds[idx] if rejecteds[idx] is not None else ""))
        
        return {
            "prompt": prompts,
            "prompt_input_ids": full_prompts_input_ids,
            "persona_attention_mask": persona_masks,
            "clean_prompt_input_ids": clean_prompts_input_ids, # added field
            "chosen": examples["chosen"],
            "rejected": rejected_responses,
            "is_dpo": is_dpo_flags,
            "has_persona": examples["has_persona"]  # added: used for PIC decision
        }

    # 5. Data loading and mixing logic
    main_process_print(f"Loading data with strategy: {script_args.mixing_strategy}")
    
    # Load raw dataset
    try:
        raw_dataset = load_dataset(
            "json",
            data_files=script_args.data_files,
            split="train"
        )  # type: ignore
    except Exception as e:
        raise RuntimeError("Failed to load dataset")
    
    # Ensure dataset is not empty
    if len(raw_dataset) == 0:  # type: ignore
        raise ValueError("Loaded dataset is empty")
    
    # Define final dataset (type determined at runtime)
    final_dataset = None

    if script_args.mixing_strategy == "original":
        # Use full dataset, keep original distribution
        main_process_print(f"Using full original dataset: {len(raw_dataset)} samples.")  # type: ignore
        final_dataset = raw_dataset.shuffle(seed=42)  # type: ignore

    elif script_args.mixing_strategy == "weighted":
        # Dynamic weighted resampling
        
        try:
            ratios = [float(r) for r in script_args.mixing_ratios.split(",")]
            if len(ratios) != 3:
                raise ValueError(f"Expected 3 ratios, got {len(ratios)}")
            total = sum(ratios)
            if total <= 0:
                raise ValueError("Ratios must sum to a positive number")
            ratios = [r/total for r in ratios]
        except (ValueError, AttributeError) as e:
            raise ValueError(f"mixing_ratios format error, expected '0.4,0.3,0.3': {e}")

        # Filter datasets by type
        ds_sft = raw_dataset.filter(  # type: ignore
            lambda x: x.get("type") == "sft" or not x.get("rejected")
        )
        
        ds_p_dpo = raw_dataset.filter(  # type: ignore
            lambda x: x.get("type") == "dpo" and x.get("has_persona") and x.get("rejected")
        )
        
        ds_n_dpo = raw_dataset.filter(  # type: ignore
            lambda x: x.get("type") == "dpo" and not x.get("has_persona") and x.get("rejected")
        )

        # Print counts for each subset
        main_process_print(f"  - SFT count: {len(ds_sft)}")  # type: ignore
        main_process_print(f"  - Persona-DPO count: {len(ds_p_dpo)}")  # type: ignore
        main_process_print(f"  - Normal-DPO count: {len(ds_n_dpo)}")  # type: ignore

        # Mix datasets
        non_empty_datasets = []
        non_empty_ratios = []
        dataset_names = []
        
        if len(ds_sft) > 0:  # type: ignore
            non_empty_datasets.append(ds_sft)
            non_empty_ratios.append(ratios[0])
            dataset_names.append("SFT")
        
        if len(ds_p_dpo) > 0:  # type: ignore
            non_empty_datasets.append(ds_p_dpo)
            non_empty_ratios.append(ratios[1])
            dataset_names.append("Persona-DPO")
        
        if len(ds_n_dpo) > 0:  # type: ignore
            non_empty_datasets.append(ds_n_dpo)
            non_empty_ratios.append(ratios[2])
            dataset_names.append("Normal-DPO")
        
        if len(non_empty_datasets) >= 2:
            ratio_sum = sum(non_empty_ratios)
            non_empty_ratios = [r / ratio_sum for r in non_empty_ratios]
            
            final_dataset = interleave_datasets(
                non_empty_datasets,
                probabilities=non_empty_ratios,
                seed=42,
                stopping_strategy=script_args.stopping_strategy
            )
            
            main_process_print(f"Weighted mixing applied for: {dataset_names}")
            main_process_print(f"Adjusted ratios: {[f'{r:.3f}' for r in non_empty_ratios]}")
            main_process_print(f"New dataset size (expanded): {len(final_dataset)}")
        
        elif len(non_empty_datasets) == 1:
            main_process_print(f"Warning: Only one dataset type available ({dataset_names[0]}). Using it directly.")
            final_dataset = non_empty_datasets[0].shuffle(seed=42)
        
        else:
            main_process_print("Error: All sub-datasets are empty. Falling back to original.")
            final_dataset = raw_dataset.shuffle(seed=42)  # type: ignore

    else:
        raise ValueError(f"Unknown mixing strategy: {script_args.mixing_strategy}")
    
    # Safety check: ensure final_dataset was initialized
    if final_dataset is None:
        raise RuntimeError("final_dataset was not properly initialized")
    
    # Use multiple processes for tokenization (limit to 8)
    num_proc = min(multiprocessing.cpu_count(), 8)
    
    main_process_print(f"Processing dataset with {num_proc} workers...")
    
    # Tokenize and process the dataset
    columns_to_remove = final_dataset.column_names if hasattr(final_dataset, 'column_names') and final_dataset.column_names else []  # type: ignore
    train_dataset = final_dataset.map(  # type: ignore
        process_and_tokenize,
        batched=True,
        num_proc=num_proc,  # type: ignore
        desc="Tokenizing and processing",  # type: ignore
        remove_columns=columns_to_remove  # type: ignore
    )
    
    main_process_print(f"Dataset processing complete: {len(train_dataset)} samples")
    
    # Reorganize dataset: ensure each global batch has fixed counts of each sample type
    def reorganize_dataset_for_fixed_batch(dataset, batch_size=None, grad_accum_steps=None, num_processes=None):
        """Form global batches with a fixed composition to ensure balanced per-GPU minibatches.

        This reordering helps DistributedSampler (no shuffle) yield balanced samples across GPUs.
        """
        if batch_size is None:
            batch_size = dpo_args.per_device_train_batch_size
        if grad_accum_steps is None:
            grad_accum_steps = dpo_args.gradient_accumulation_steps
        if num_processes is None:
            num_processes = distributed_state.num_processes

        per_device_accumulation_size = batch_size * grad_accum_steps
        global_batch_size = per_device_accumulation_size * num_processes

        sft_samples = [item for item in dataset if not item["is_dpo"]]
        persona_dpo_samples = [item for item in dataset if item["is_dpo"] and item["has_persona"]]
        normal_dpo_samples = [item for item in dataset if item["is_dpo"] and not item["has_persona"]]

        main_process_print(f"\nReorganizing dataset:")
        main_process_print(f"  Config: per_device_batch={batch_size} × grad_accum={grad_accum_steps} × {num_processes}GPUs = global_batch={global_batch_size}")
        main_process_print(f"  - SFT samples: {len(sft_samples)}")
        main_process_print(f"  - Persona-DPO samples: {len(persona_dpo_samples)}")
        main_process_print(f"  - Normal-DPO samples: {len(normal_dpo_samples)}")

        total_samples = len(sft_samples) + len(persona_dpo_samples) + len(normal_dpo_samples)
        sft_ratio = len(sft_samples) / total_samples if total_samples > 0 else 0
        persona_dpo_ratio = len(persona_dpo_samples) / total_samples if total_samples > 0 else 0
        normal_dpo_ratio = len(normal_dpo_samples) / total_samples if total_samples > 0 else 0

        if global_batch_size >= 3 and total_samples > 0:
            n_sft = int(global_batch_size * sft_ratio)
            n_persona_dpo = int(global_batch_size * persona_dpo_ratio)
            n_normal_dpo = int(global_batch_size * normal_dpo_ratio)

            if len(sft_samples) > 0 and n_sft == 0:
                n_sft = 1
            if len(persona_dpo_samples) > 0 and n_persona_dpo == 0:
                n_persona_dpo = 1
            if len(normal_dpo_samples) > 0 and n_normal_dpo == 0:
                n_normal_dpo = 1

            current_sum = n_sft + n_persona_dpo + n_normal_dpo
            diff = global_batch_size - current_sum

            if diff > 0:
                sft_frac = global_batch_size * sft_ratio - n_sft
                persona_dpo_frac = global_batch_size * persona_dpo_ratio - n_persona_dpo
                normal_dpo_frac = global_batch_size * normal_dpo_ratio - n_normal_dpo

                adjustments = [
                    (sft_frac, 'sft'),
                    (persona_dpo_frac, 'persona_dpo'),
                    (normal_dpo_frac, 'normal_dpo')
                ]
                adjustments.sort(reverse=True)

                for i in range(diff):
                    if i < len(adjustments):
                        if adjustments[i][1] == 'sft':
                            n_sft += 1
                        elif adjustments[i][1] == 'persona_dpo':
                            n_persona_dpo += 1
                        elif adjustments[i][1] == 'normal_dpo':
                            n_normal_dpo += 1
            elif diff < 0:
                main_process_print(f"  ⚠ Warning: computed sample total ({current_sum}) exceeds global_batch_size ({global_batch_size}), adjusting")
                if n_sft >= n_persona_dpo and n_sft >= n_normal_dpo:
                    n_sft += diff
                elif n_persona_dpo >= n_normal_dpo:
                    n_persona_dpo += diff
                else:
                    n_normal_dpo += diff
        else:
            main_process_print(f"  ⚠ Warning: global_batch_size={global_batch_size} < 3 or dataset empty; cannot balance three types")
            n_persona_dpo = global_batch_size
            n_normal_dpo = 0
            n_sft = 0

        main_process_print(f"  Global batch composition: {n_persona_dpo} Persona-DPO + {n_normal_dpo} Normal-DPO + {n_sft} SFT")
        main_process_print(f"  Interleaving applied to accommodate DistributedSampler stride allocation")

        max_batches = min(
            len(persona_dpo_samples) // n_persona_dpo if n_persona_dpo > 0 else float('inf'),
            len(normal_dpo_samples) // n_normal_dpo if n_normal_dpo > 0 else float('inf'),
            len(sft_samples) // n_sft if n_sft > 0 else float('inf')
        )

        if max_batches == float('inf'):
            max_batches = max(
                len(persona_dpo_samples) // n_persona_dpo if n_persona_dpo > 0 else 0,
                len(normal_dpo_samples) // n_normal_dpo if n_normal_dpo > 0 else 0,
                len(sft_samples) // n_sft if n_sft > 0 else 0
            )

        main_process_print(f"  Formable full global batches: {max_batches}")

        max_batches = int(max_batches)

        reorganized = []
        for batch_idx in range(max_batches):
            current_batch_samples = []

            if n_persona_dpo > 0:
                start_idx = batch_idx * n_persona_dpo
                current_batch_samples.extend(persona_dpo_samples[start_idx:start_idx + n_persona_dpo])

            if n_normal_dpo > 0:
                start_idx = batch_idx * n_normal_dpo
                current_batch_samples.extend(normal_dpo_samples[start_idx:start_idx + n_normal_dpo])

            if n_sft > 0:
                start_idx = batch_idx * n_sft
                current_batch_samples.extend(sft_samples[start_idx:start_idx + n_sft])

            interleaved_batch = []
            batch_len = len(current_batch_samples)

            samples_per_gpu = (batch_len + num_processes - 1) // num_processes

            for gpu_idx in range(num_processes):
                for local_idx in range(samples_per_gpu):
                    global_idx = gpu_idx * samples_per_gpu + local_idx
                    if global_idx < batch_len:
                        interleaved_batch.append(current_batch_samples[global_idx])

            reorganized.extend(interleaved_batch)

        main_process_print(f"  Reorganized dataset size: {len(reorganized)} samples")
        main_process_print(f"  (Interleaved for balanced per-GPU distribution)\n")

        from datasets import Dataset as HFDataset
        return HFDataset.from_list(reorganized)
    
    # Apply reorganization (used to enforce fixed global-batch composition)
    train_dataset = reorganize_dataset_for_fixed_batch(train_dataset)
    
    # Summary statistics for training dataset
    main_process_print("\n" + "="*60)
    main_process_print("Training dataset statistics")
    main_process_print("="*60)
    
    sft_count = sum(1 for item in train_dataset if not item["is_dpo"])  # type: ignore
    dpo_with_persona_count = sum(1 for item in train_dataset if item["is_dpo"] and item["has_persona"])  # type: ignore
    dpo_without_persona_count = sum(1 for item in train_dataset if item["is_dpo"] and not item["has_persona"])  # type: ignore
    
    # verify generation of clean_prompt_input_ids
    pic_with_clean_ids = sum(1 for item in train_dataset if item["is_dpo"] and item["has_persona"] and len(item.get("clean_prompt_input_ids", [])) > 0)  # type: ignore
    pic_without_clean_ids = sum(1 for item in train_dataset if item["is_dpo"] and item["has_persona"] and len(item.get("clean_prompt_input_ids", [])) == 0)  # type: ignore
    
    main_process_print(f"Total samples: {len(train_dataset)}")
    main_process_print(f"  ├─ Total SFT samples: {sft_count}")
    main_process_print(f"  └─ Total DPO samples: {dpo_with_persona_count + dpo_without_persona_count}")
    main_process_print(f"      ├─ Persona-DPO: {dpo_with_persona_count}")
    main_process_print(f"      │   ├─ ✓ with clean_prompt_ids: {pic_with_clean_ids} (PIC computable)")
    main_process_print(f"      │   └─ ✗ without clean_prompt_ids: {pic_without_clean_ids} (PIC skipped)")
    main_process_print(f"      └─ Normal-DPO: {dpo_without_persona_count}")
    main_process_print("="*60 + "\n")
    
    # Check whether persona-DPO samples have clean_prompt_input_ids
    if pic_without_clean_ids > 0:
        main_process_print(f"⚠ Warning: {pic_without_clean_ids} persona-DPO samples are missing clean_prompt_input_ids. PIC will be skipped for them.")
        main_process_print("  These samples cannot contribute to PIC loss. Please inspect preprocessing.\n")
    else:
        main_process_print("✓ All persona-DPO samples generated clean_prompt_input_ids\n")

    # 6. Model loading

    # build quantization config
    quantization_config = get_quantization_config(model_config)
    
    # define model loading kwargs
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch.float16,
        use_cache=False,
    )
    
    # handle quantization config and device mapping
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        if distributed_state.num_processes > 1:
            model_kwargs["device_map"] = {"": distributed_state.process_index}
        else:
            model_kwargs["device_map"] = "auto"
    
    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs
    )

    # 7. Prepare model for k-bit/quantized training
    if model_config.load_in_4bit or model_config.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=dpo_args.gradient_checkpointing
        )

    # 8. Get LoRA (PEFT) config
    peft_config = get_peft_config(model_config)

    # 9. Initialize Robust DPO trainer
    trainer = RobustDPOTrainer(
        model=model,
        ref_model=None,
        peft_config=peft_config,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=RobustDataCollator(tokenizer.pad_token_id),
    )

    # 10. Print training configuration
    main_process_print("\n" + "="*60)
    main_process_print("Training Configuration Summary")
    main_process_print("="*60)
    main_process_print(f"DPO Loss: Enabled (beta={dpo_args.beta})")
    main_process_print(f"SFT Loss: {'Enabled' if dpo_args.sft_alpha > 0 else 'Disabled'} (alpha={dpo_args.sft_alpha})")
    main_process_print(f"PIC Loss: {'Enabled' if dpo_args.pic_lambda > 0 else 'Disabled'} (lambda={dpo_args.pic_lambda}, top_k={dpo_args.pic_top_k})")
    main_process_print("="*60 + "\n")
    
    # 11. Start training
    main_process_print("Starting Training with DPO+SFT+PIC...")
    trainer.train()
    trainer.save_model(dpo_args.output_dir)
    main_process_print("Model saved [REDACTED]")

    # Distributed cleanup: attempt to destroy NCCL/process group to avoid warnings
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # wait for all processes to reach this barrier, then destroy the process group
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            torch.distributed.destroy_process_group()
            main_process_print("torch.distributed process group destroyed.")
    except Exception as e:
        main_process_print(f"Warning: failed to destroy process group: {e}")

if __name__ == "__main__":
    main()