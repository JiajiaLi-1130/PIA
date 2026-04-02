"""
Dynamic instructions - persona evolution (mixed datasets)
Example:
python src/PLE/mixdata_attack.py \
    --config src/PLE/attack_config.yaml \
    --harm_file_path_A data/JBB-Behaviors-harmful.jsonl \
    --harm_file_path_B data/PKU-SafeRLHF-Train-unsafe.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

# --- import components from attack.py ---
try:
    import attack
    from attack import _HAS_VLLM, Config, PerformanceMonitor, PersonaEvolutionGraph, read_personas_file, read_prompts_file
except ImportError:
    print("Error: failed to import attack.py. Ensure this script is located with attack.py.")
    sys.exit(1)

from attack import logger


def read_harmful_prompts_with_metadata(path: str) -> list[dict]:
    data_list: list[dict] = []
    path_str = str(path)
    prompt_fields = ['prompt', 'text', 'input', 'query']  # support multiple prompt field names
    try:
        if path_str.endswith('.jsonl') or path_str.endswith('.jl'):
            # read JSONL file
            with open(path_str, encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            # check if any prompt field is present
                            has_prompt = any(field in data for field in prompt_fields)
                            if has_prompt:
                                data_list.append(data)
                            else:
                                logger.warning(f"{path_str} line {i} missing prompt fields ({', '.join(prompt_fields)})")
                        else:
                            logger.warning(f"{path_str} line {i} is not a valid JSON object")
                    except json.JSONDecodeError as e:
                        logger.warning(f"{path_str} line {i} failed to parse: {e}")
        else:
            # read JSON file
            with open(path_str, encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # check if any prompt field is present
                        has_prompt = any(field in item for field in prompt_fields)
                        if has_prompt:
                            data_list.append(item)
                        else:
                            logger.warning(f"Item in {path_str} missing prompt fields ({', '.join(prompt_fields)})")
                    else:
                        logger.warning(f"Item in {path_str} is not a valid JSON object")
            elif isinstance(data, dict):
                # attempt to extract prompt list from single dict
                for field in prompt_fields:
                    if field in data and isinstance(data[field], list):
                        for item in data[field]:
                            if isinstance(item, dict):
                                data_list.append(item)
                        break
                else:
                    logger.warning(f"{path_str} does not contain a valid prompt list")
            else:
                logger.error(f"{path_str} is not valid JSON format")
                
        logger.info(f"Loaded {len(data_list)} harmful prompts from {path_str}")
        return data_list
    except Exception as e:
        logger.error(f"Error reading {path_str}: {e}")
        return []

def get_prompt_from_item(item: dict, prompt_fields: tuple[str, ...] = ('prompt', 'text', 'input', 'query')) -> str | None:
    for field in prompt_fields:
        if field in item and isinstance(item[field], str):
            return item[field]
    return None

class DynamicEvolutionGraph(PersonaEvolutionGraph):
    """Extension of PersonaEvolutionGraph that supports mixed-dataset evaluation.

    Strategy: full fixed set A + global non-replacement sampled pool B
    (slice allocation per generation).
    """
    def __init__(self, 
                 initial_personas: list[str],
                 harmful_prompts: list[Any],
                 inference_config: dict,
                 judger_config: dict,
                 api_generator_config: dict,
                 fixed_dataset_A: list[Any] | None = None,
                 pool_dataset_B: list[Any] | None = None,
                 pool_dataset_B_with_meta: list[dict] | None = None,
                 sample_size: int = 50,
                 beta: float = 0.3,
                 epsilon: float = 0.1,
                 ucb_c: float = 1.0,
                 selection_strategy: str = 'ucb',
                 evaluate_metric: str = 'asr',
                 parent_to_children: int = 1,
                 select_parents_num: int = 16,
                 monitor: PerformanceMonitor | None = None,
                 output_dir: Path | str | None = None):
        # initialize parent class; keep parameter order consistent
        super().__init__(
            initial_personas=initial_personas,
            harmful_prompts=harmful_prompts,
            inference_config=inference_config,
            api_generator_config=api_generator_config,
            judger_config=judger_config,
            beta=beta,
            epsilon=epsilon,
            ucb_c=ucb_c,
            selection_strategy=selection_strategy,
            evaluate_metric=evaluate_metric,
            parent_to_children=parent_to_children,
            select_parents_num=select_parents_num,
            monitor=monitor,
            output_dir=output_dir
        )
        
        # save parameters and datasets
        self.sample_size = sample_size
        self.fixed_dataset_A = fixed_dataset_A if fixed_dataset_A else []
        self.pool_dataset_B = pool_dataset_B if pool_dataset_B else []
        self.pool_dataset_B_with_meta = pool_dataset_B_with_meta if pool_dataset_B_with_meta else []
        # store full metadata for fixed set A (provided externally)
        self.fixed_dataset_A_with_meta = []
        
        # add a tracker for global non-replacement sampling
        self.current_sample_start = 0

        # validate datasets
        if not self.fixed_dataset_A:
            logger.warning("Warning: fixed dataset A is empty")
        if not self.pool_dataset_B:
            logger.warning("Warning: pool dataset B is empty")
        if not self.pool_dataset_B_with_meta:
            logger.warning("Warning: pool dataset B metadata is empty")

        # check if pool B size is sufficient for sampling needs
        total_needed_samples = sample_size * 100
        if len(self.pool_dataset_B_with_meta) < total_needed_samples:
            logger.warning(f"Warning: pool B size ({len(self.pool_dataset_B_with_meta)}) may be insufficient for non-replacement sampling (per-gen {sample_size})")

        logger.info(f"Mixed strategy initialized: fixed A={len(self.fixed_dataset_A)}, pool B={len(self.pool_dataset_B)}, sample_size={self.sample_size}")

    async def evaluate_and_update(self, node_ids: list[str], generation_count: int = 0, prompts: list[str] | None = None):
        """Assemble self.harmful_prompts dynamically before calling parent logic."""
        # 1. perform global non-replacement sampling (slice allocation)
        sampled_B_with_meta = []
        
        try:
            # compute slice range for current generation
            start_idx = self.current_sample_start
            end_idx = start_idx + self.sample_size

            # get samples for current slice
            if start_idx < len(self.pool_dataset_B_with_meta):
                # ensure not to exceed dataset bounds
                actual_end_idx = min(end_idx, len(self.pool_dataset_B_with_meta))
                sampled_B_with_meta = self.pool_dataset_B_with_meta[start_idx:actual_end_idx]

                # update start index for next sampling
                self.current_sample_start = actual_end_idx

                # keep items with metadata that provide prompt text
                sampled_B_with_meta = [item for item in sampled_B_with_meta if get_prompt_from_item(item) is not None]
            else:
                logger.warning("Reached end of pool dataset B; no more samples available")
            
            if len(sampled_B_with_meta) == 0:
                logger.warning("No valid samples extracted from pool B; will use fixed set A only")
                sampled_B_with_meta = []
        except Exception as e:
            logger.error(f"Error sampling from pool B: {e}; using fixed set A only")
            sampled_B_with_meta = []
        
        # 2. assemble prompt set (fixed A + sampled B) - keep full metadata dicts
        mixed_prompts = self.fixed_dataset_A_with_meta + sampled_B_with_meta

        # update parent class prompt list
        self.harmful_prompts = mixed_prompts
        
        logger.info(f"Evaluation prompt composition: fixed A({len(self.fixed_dataset_A_with_meta)}) + sampled B({len(sampled_B_with_meta)}) = total {len(self.harmful_prompts)}")
        logger.info(f"Sampling window: generation {generation_count+1}, pool B index [{start_idx}-{end_idx}), actual sampled {len(sampled_B_with_meta)}")
        
        # 3. call parent logic to perform inference and scoring
        try:
            await super().evaluate_and_update(node_ids, generation_count)
        except Exception as e:
            logger.error(f"Error evaluating and updating nodes: {e}")
            raise

async def async_main_mixed():
    """Custom main to handle two datasets (fixed A + pool B)."""
    # initialize performance monitor
    perf_monitor = PerformanceMonitor()
    perf_monitor.start_timer()

    try:
        # --- 1. Argument parsing ---
        parser = argparse.ArgumentParser(description='Dynamic persona evolution (Fixed A + Random B)')
        
        parser.add_argument('--config', type=str, help='Path to configuration file')
        parser.add_argument('--harm_file_path_A', type=str, required=True, help='Fixed harmful prompt set A (overrides harm_file_path in config)')
        parser.add_argument('--harm_file_path_B', type=str, required=True, help='Path to pool dataset B for sampling')
        parser.add_argument('--output_path', type=str, help='Output file path')
        parser.add_argument('--persona_path', type=str, help='Persona data file path')
        parser.add_argument('--generations', type=int, help='Number of generations')
        parser.add_argument('--epsilon', type=float, help='Exploration probability')
        parser.add_argument('--beta', type=float, help='Propagation decay factor')
        parser.add_argument('--parent_to_children', type=int, default=1, help='Number of children per parent')
        parser.add_argument('--select_parents_num', type=int, default=10, help='Number of parents to select')
        parser.add_argument('--eval_batch_size', type=int, default=5, help='Evaluation batch size')
        parser.add_argument('--ucb_c', type=float, help='UCB exploration coefficient (smaller = more conservative)')
        parser.add_argument('--selection_strategy', type=str, choices=['ucb', 'epsilon', 'hybrid'], help='Parent selection strategy (ucb/epsilon/hybrid)')
        parser.add_argument('--evaluate_metric', type=str, choices=['asr', 'rta'], help='Evaluation metric (asr: attack success rate, rta: rejection rate)')
        parser.add_argument('--asr_threshold', type=float, default=0.6, help='ASR threshold')
        parser.add_argument('--elite_N', type=int, default=35, help='Number of elite personas')
        # API generator args
        parser.add_argument('--api_key', type=str, default=None, help='External API key')
        parser.add_argument('--api_base_url', type=str, help='External API base URL')
        parser.add_argument('--api_model', type=str, help='External API model identifier')
        parser.add_argument('--api_max_concurrency', type=int  , help='API max concurrency')
        
        # inference model args
        parser.add_argument('--inference_max_model_len', type=int, help='Inference max input tokens')
        parser.add_argument('--inference_max_output_tokens', type=int, help='Inference max output tokens')
        parser.add_argument('--inference_batch_size', type=int, help='Inference batch size')
        parser.add_argument('--inference_devices', type=str, help='Visible GPUs for inference')
        parser.add_argument('--inference_gpu_memory_utilization', type=float, help='Inference GPU memory utilization (float)')

        # judger model args
        parser.add_argument('--judger_max_model_len', type=int, help='Judger max input tokens')
        parser.add_argument('--judger_max_output_tokens', type=int, help='Judger max output tokens')
        parser.add_argument('--judger_batch_size', type=int, help='Judger batch size')
        parser.add_argument('--judger_devices', type=str, help='Visible GPUs for judger')
        parser.add_argument('--judger_gpu_memory_utilization', type=float, help='Judger GPU memory utilization (float)')
        parser.add_argument('--tensor_parallel_size', type=int, help='Tensor parallel size')
        
        # log level
        parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Log level')
        
        # mixed-strategy args
        parser.add_argument('--sample_size', type=int, default=50, help='Number of samples to draw from pool B per generation')
        
        # Configuration precedence: CLI > YAML file > defaults in code
        # Load YAML (if exists), then parse CLI and let CLI values (non-None) override file values.
        # The merged flat dict is passed to attack.Config.from_dict
        config_arg, remaining_args = parser.parse_known_args()
        config_file = config_arg.config if config_arg.config else str(Path(__file__).parent / "attack.yaml")

        # Load configuration from file (returns dict, may contain Path objects)
        file_config = attack.load_config(config_file) or {}

        # parse full CLI args (CLI overrides file_config)
        args = parser.parse_args()

        # set log level
        logger.setLevel(getattr(logging, args.log_level.upper()))

        # build flattened CLI config dict; keep non-None values to override YAML
        cli_config = {k: v for k, v in vars(args).items() if v is not None}

        # merge: start from file_config, then override with CLI
        merged_config = {}
        merged_config.update(file_config)
        merged_config.update(cli_config)

        # info: list only keys to avoid leaking secrets; switch to debug if needed
        logger.info(f"Loaded configuration file: {config_file}")
        logger.debug(f"Merged config keys: {list(merged_config.keys())}")

        # build Config object
        try:
            config = Config.from_dict(merged_config)
            logger.info("Configuration validated")
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Configuration validation failed: {e}")
            return
    except Exception as e:
        logger.error(f"Error during argument parsing and config loading: {e}")
        return

    if not _HAS_VLLM:
        logger.error("vLLM is not installed")
        return

    # --- 2. Load datasets ---
    logger.info("Loading datasets...")
    
    # load personas (seed)
    initial_personas = read_personas_file(config.paths.persona_path)
    if not initial_personas:
        logger.error("Failed to load initial persona data")
        return
    logger.info(f"Loaded initial personas: {len(initial_personas)} entries")
    
    # load fixed set A (keep metadata)
    dataset_A_with_meta = read_harmful_prompts_with_metadata(args.harm_file_path_A)
    if not dataset_A_with_meta:
        logger.error(f"Failed to load any valid metadata from {args.harm_file_path_A}")
        return
    dataset_A = dataset_A_with_meta
    logger.info(f"Loaded fixed dataset A: {len(dataset_A)} entries (with metadata)")
    
    # load prompt set B (pool)
    # For pool B, keep full metadata to support balanced sampling by harm category
    try:
        dataset_B_with_meta = read_harmful_prompts_with_metadata(args.harm_file_path_B)
        if not dataset_B_with_meta:
            logger.error(f"Failed to load any valid metadata from {args.harm_file_path_B}")
            return
        logger.info(f"Loaded pool dataset B: {len(dataset_B_with_meta)} entries (with metadata)")
    except Exception as e:
        logger.error(f"Failed to load pool dataset B: {e}")
        return

    # --- 3. Hardware configuration ---
    try:
        # get device configuration
        inference_devices = config.inference.devices
        judger_devices = config.judger.devices
        
        if not inference_devices or not judger_devices:
            raise ValueError("inference_devices and judger_devices must be specified in the configuration.")
        
        # compute tensor-parallel size from devices list
        inference_tp_size = len(inference_devices.split(','))
        judger_tp_size = len(judger_devices.split(','))
        
        if inference_tp_size == 0 or judger_tp_size == 0:
            raise ValueError("devices string must not be empty.")

        logger.info(f"Inference -> GPUs [{inference_devices}] (TP={inference_tp_size})")
        logger.info(f"Judger -> GPUs [{judger_devices}] (TP={judger_tp_size})")

    except Exception as e:
        logger.error(f"GPU allocation strategy failed: {e}")
        return
    
    # inference model config
    inference_config = {
        "model_dir": config.inference.model_dir,
        "tensor_parallel_size": inference_tp_size,
        "max_model_len": config.inference.max_model_len,
        "max_output_tokens": config.inference.max_output_tokens,
        "batch_size": config.inference.batch_size,
        "visible_devices": inference_devices,
        "gpu_memory_utilization": getattr(config.inference, 'gpu_memory_utilization', 0.8)
    }
    
    # judger model config
    judger_config = {
        "model_dir": config.judger.model_dir,
        "tensor_parallel_size": judger_tp_size,
        "max_model_len": config.judger.max_model_len,
        "max_output_tokens": config.judger.max_output_tokens,
        "batch_size": config.judger.batch_size,
        "visible_devices": judger_devices,
        "gpu_memory_utilization": getattr(config.judger, 'gpu_memory_utilization', 0.9)
    }
    
    # API generator config
    api_config = {
        "model": config.api_generator.model,
        "api_key": config.api_generator.api_key,
        "base_url": config.api_generator.base_url,
        "max_concurrency": config.api_generator.max_concurrency,
    }

    # --- 4. Initialize mixed-strategy graph (DynamicEvolutionGraph) ---
    logger.info("Initializing mixed-strategy evolution graph...")
    
    try:
        # Build plaintext views expected by the parent while preserving original metadata
        _tmp_harm = [get_prompt_from_item(item) for item in dataset_A]
        harmful_prompts_init = [p for p in _tmp_harm if p is not None]
        harmful_prompts_init = cast(list[str], harmful_prompts_init)

        _tmp_pool = [get_prompt_from_item(item) for item in dataset_B_with_meta]
        pool_B_prompts = [p for p in _tmp_pool if p is not None]
        pool_B_prompts = cast(list[str], pool_B_prompts)

        # construct plaintext view of fixed set A while keeping metadata
        _tmp_fixed_A = [get_prompt_from_item(item) for item in dataset_A]
        fixed_A_prompts = [p for p in _tmp_fixed_A if p is not None]
        fixed_A_prompts = cast(list[str], fixed_A_prompts)

        peg = DynamicEvolutionGraph(
            initial_personas=initial_personas,
            harmful_prompts=harmful_prompts_init,
            fixed_dataset_A=fixed_A_prompts,
            pool_dataset_B=pool_B_prompts,
            pool_dataset_B_with_meta=dataset_B_with_meta,
            sample_size=args.sample_size,
            inference_config=inference_config,
            judger_config=judger_config,
            api_generator_config=api_config,
            monitor=perf_monitor,
            beta=config.evolution.beta,
            epsilon=config.evolution.epsilon,
            ucb_c=config.evolution.ucb_c,
            selection_strategy=config.evolution.selection_strategy,
            evaluate_metric=config.evolution.evaluate_metric,
            parent_to_children=config.evolution.parent_to_children,
            select_parents_num=config.evolution.select_parents_num,
            output_dir=config.paths.output_path.parent
        )
        # set full metadata for fixed set A
        peg.fixed_dataset_A_with_meta = dataset_A
    except Exception as e:
        logger.error(f"Failed to initialize mixed-strategy evolution graph: {e}")
        return

    # --- 5. Start evolution ---
    logger.info("Starting mixed-strategy evolution...")
    try:
        await peg.evolve(
            generations=config.evolution.generations,
            asr_threshold=config.evolution.asr_threshold,
            elite_N=config.evolution.elite_N,
            select_parents_num=config.evolution.select_parents_num,
            eval_batch_size=config.evolution.eval_batch_size
        )
    except Exception as e:
        logger.error(f"Error during evolution: {e}")
        return

    # --- 6. Cleanup ---
    # simple memory cleanup (refer to attack.py)
    import gc
    del peg
    gc.collect()

if __name__ == "__main__":
    try:
        asyncio.run(async_main_mixed())
    except Exception:
        logger.exception("Asynchronous main process failed")
        sys.exit(1)