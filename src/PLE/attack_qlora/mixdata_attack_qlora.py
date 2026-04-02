"""
PLE - Dynamic prompts (mixed datasets)

Example run:
python src/PLE/attack_qlora/mixdata_attack_qlora.py \
    --config src/PLE/attack_qlora/attack_qlora.yaml \
    --harm_file_path_A data/JBB-Behaviors-harmful.jsonl \
    --harm_file_path_B data/PKU-SafeRLHF-Train-unsafe.jsonl
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

# --- import components from attack_qlora.py ---
try:
    import attack_qlora
    from attack_qlora import _HAS_VLLM, Config, PerformanceMonitor, PersonaEvolutionGraph, read_personas_file, read_prompts_file
except ImportError:
    print("Error: failed to import attack_qlora.py. Ensure this script is in the same directory as attack_qlora.py.")
    sys.exit(1)

from attack_qlora import logger


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
                            # check for any prompt fields
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
                        # check for any prompt fields
                        has_prompt = any(field in item for field in prompt_fields)
                        if has_prompt:
                            data_list.append(item)
                        else:
                            logger.warning(f"Items in {path_str} missing prompt fields ({', '.join(prompt_fields)})")
                    else:
                        logger.warning(f"Items in {path_str} are not valid JSON objects")
            elif isinstance(data, dict):
                for field in prompt_fields:
                    if field in data and isinstance(data[field], list):
                        for item in data[field]:
                            if isinstance(item, dict):
                                data_list.append(item)
                        break
                else:
                    logger.warning(f"{path_str} does not contain a valid prompts list")
            else:
                logger.error(f"{path_str} is not valid JSON format")
                
        logger.info(f"Successfully read {len(data_list)} harmful prompts from {path_str}")
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
    """
    Subclass of the evolution graph that overrides evaluation logic
    to support a mixed prompt strategy.
    Strategy: full fixed set A + global non-replacement sampling from pool B (slice per generation).
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
        # Initialize parent class; ensure parameter order matches parent
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
        
        # Save parameters and datasets
        self.sample_size = sample_size
        self.fixed_dataset_A = fixed_dataset_A if fixed_dataset_A else []
        self.pool_dataset_B = pool_dataset_B if pool_dataset_B else []
        self.pool_dataset_B_with_meta = pool_dataset_B_with_meta if pool_dataset_B_with_meta else []
        # Reserve a slot for full metadata of fixed set A (provided externally)
        self.fixed_dataset_A_with_meta = []
        
        # Add position tracking for global non-replacement sampling
        self.current_sample_start = 0
        
        # validate datasets
        if not self.fixed_dataset_A:
            logger.warning("Warning: fixed dataset A is empty!")
        if not self.pool_dataset_B:
            logger.warning("Warning: pool dataset B is empty!")
        if not self.pool_dataset_B_with_meta:
            logger.warning("Warning: pool dataset B metadata is empty!")
        
        # check whether pool B is large enough for sampling needs
        total_needed_samples = sample_size * 100  # assume up to 100 generations
        if len(self.pool_dataset_B_with_meta) < total_needed_samples:
            logger.warning(f"Warning: pool B size ({len(self.pool_dataset_B_with_meta)}) may be insufficient for global non-replacement sampling (sample_size={sample_size} per generation)")

        logger.info(f"Mixed-strategy initialized: |fixed A|={len(self.fixed_dataset_A)}, |pool B|={len(self.pool_dataset_B)}, sample_size={self.sample_size}")

    async def evaluate_and_update(self, node_ids: list[str], generation_count: int = 0, prompts: list[str] | None = None):
        """
        Assemble self.harmful_prompts dynamically before calling parent logic.
        """
        # 1. perform global non-replacement sampling (slice-based)
        sampled_B_with_meta = []
        
        try:
            # compute slice indices for the current generation
            start_idx = self.current_sample_start
            end_idx = start_idx + self.sample_size

            # get the current slice
            if start_idx < len(self.pool_dataset_B_with_meta):
                actual_end_idx = min(end_idx, len(self.pool_dataset_B_with_meta))
                sampled_B_with_meta = self.pool_dataset_B_with_meta[start_idx:actual_end_idx]

                # update next start
                self.current_sample_start = actual_end_idx

                # keep only items that yield a prompt string
                sampled_B_with_meta = [item for item in sampled_B_with_meta if get_prompt_from_item(item) is not None]
            else:
                logger.warning("Reached the end of pool dataset B; no more new samples can be drawn")

            if len(sampled_B_with_meta) == 0:
                logger.warning("No valid samples were drawn from pool B; will use fixed dataset A only")
                sampled_B_with_meta = []
        except Exception as e:
            logger.error(f"Error sampling from pool B: {e}; will use fixed dataset A only")
            sampled_B_with_meta = []
        
        # 2. assemble prompts (fixed A + sampled B) - keep full metadata dicts
        mixed_prompts = self.fixed_dataset_A_with_meta + sampled_B_with_meta

        # update the prompt list used by the parent class
        self.harmful_prompts = mixed_prompts

        logger.info(f"This evaluation uses: fixed A ({len(self.fixed_dataset_A_with_meta)}) + sampled B ({len(sampled_B_with_meta)}) = total {len(self.harmful_prompts)} prompts")
        logger.info(f"Sampling window: generation {generation_count+1}, pool B index [{start_idx}-{end_idx}), drawn {len(sampled_B_with_meta)} items")
        
        # 3. call parent logic to perform inference and scoring
        try:
            await super().evaluate_and_update(node_ids, generation_count)
        except Exception as e:
            logger.error(f"Error evaluating and updating nodes: {e}")
            raise

async def async_main_mixed():
    """
    Custom main function that handles loading two datasets (fixed A + pool B).
    """
    # initialize performance monitor
    perf_monitor = PerformanceMonitor()
    perf_monitor.start_timer()

    try:
        # --- 1. argument parsing ---
        parser = argparse.ArgumentParser(description='Dynamic persona evolution (Fixed A + Random B)')

        parser.add_argument('--config', type=str, help='Path to configuration file')
        parser.add_argument('--harm_file_path_A', type=str, required=True, help='Fixed harmful prompt set A')
        parser.add_argument('--harm_file_path_B', type=str, required=True, help='Pool B path for random sampling')
        parser.add_argument('--output_path', type=str, help='Output path')
        parser.add_argument('--persona_path', type=str, help='Persona data file path')
        parser.add_argument('--generations', type=int, help='Number of generations')
        parser.add_argument('--beta', type=float, help='Propagation decay factor')
        parser.add_argument('--parent_to_children', type=int, default=1, help='Children per parent')
        parser.add_argument('--select_parents_num', type=int, default=10, help='Number of parents to select')
        parser.add_argument('--eval_batch_size', type=int, default=5, help='Evaluation batch size')
        parser.add_argument('--ucb_c', type=float, help='UCB exploration coefficient (smaller => more conservative)')
        parser.add_argument('--selection_strategy', type=str, choices=['ucb', 'epsilon', 'hybrid'], help='Parent selection strategy')
        parser.add_argument('--evaluate_metric', type=str, choices=['asr', 'rta'], help='Evaluation metric (asr: attack success rate, rta: refusal rate)')
        parser.add_argument('--asr_threshold', type=float, default=0.6, help='ASR threshold')
        parser.add_argument('--elite_N', type=int, default=35, help='Number of elite personas')
        parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration probability')
        # API generator args
        parser.add_argument('--api_key', type=str, default=None, help='Dashscope API key')
        parser.add_argument('--api_base_url', type=str, help='Dashscope API base URL')
        parser.add_argument('--api_model', type=str, help='Dashscope model for evolution')
        parser.add_argument('--api_max_concurrency', type=int, help='API concurrency limit')

        # inference model args
        parser.add_argument('--inference_max_model_len', type=int)
        parser.add_argument('--inference_max_output_tokens', type=int, help='Max output tokens for inference')
        parser.add_argument('--inference_batch_size', type=int, help='Inference batch size')
        parser.add_argument('--inference_devices', type=str, help='Visible GPUs for inference')
        parser.add_argument('--inference_gpu_memory_utilization', type=float, help='Inference GPU memory utilization')

        # safety judger args
        parser.add_argument('--judger_max_model_len', type=int)
        parser.add_argument('--judger_max_output_tokens', type=int, help='Max output tokens for judger')
        parser.add_argument('--judger_batch_size', type=int, help='Judger batch size')
        parser.add_argument('--judger_devices', type=str, help='Visible GPUs for judger')
        parser.add_argument('--judger_gpu_memory_utilization', type=float, help='Judger GPU memory utilization')
        parser.add_argument('--tensor_parallel_size', type=int, help='Tensor parallel size')

        # logging level
        parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')

        # mixing strategy args
        parser.add_argument('--sample_size', type=int, default=50, help='Number of samples to draw from dataset B')
        
        config_arg, remaining_args = parser.parse_known_args()
        config_file = config_arg.config if config_arg.config else str(Path(__file__).parent / "attack_qlora.yaml")

        file_config = attack_qlora.load_config(config_file) or {}

        args = parser.parse_args()

        logger.setLevel(getattr(logging, args.log_level.upper()))

        cli_config = {k: v for k, v in vars(args).items() if v is not None}

        merged_config = {}
        merged_config.update(file_config)
        merged_config.update(cli_config)

        # Debug output (list keys only to avoid leaking secrets); use logger.debug if needed
        logger.info(f"Loading config file: {config_file}")
        logger.debug(f"Merged config keys: {list(merged_config.keys())}")

        try:
            config = Config.from_dict(merged_config)
            logger.info("Config validated successfully")
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Config validation failed: {e}")
            return
    except Exception as e:
        logger.error(f"Error during argument parsing and config loading: {e}")
        return

    if not _HAS_VLLM:
        logger.error("vLLM is not installed")
        return
    # --- 2. load datasets ---
    logger.info("Loading datasets...")
    
    # load personas (seed)
    initial_personas = read_personas_file(config.paths.persona_path)
    # if file uses 'prompt' field instead of 'persona', fall back to prompts loader
    if not initial_personas:
        logger.error("Failed to load initial personas")
        return
    logger.info(f"Loaded initial personas: {len(initial_personas)} items")
    
    # load fixed set A (keep metadata)
    dataset_A_with_meta = read_harmful_prompts_with_metadata(args.harm_file_path_A)
    if not dataset_A_with_meta:
        logger.error(f"Failed to load any valid metadata from {args.harm_file_path_A}")
        return
    # dataset_A may be a list of dicts with metadata; keep original structure so outputs can include chosen/rejected
    dataset_A = dataset_A_with_meta
    logger.info(f"Loaded fixed set A: {len(dataset_A)} items (with metadata)")
    
    # load prompt set B (pool)
    # For pool B we need to keep full metadata to support balanced sampling by harm category
    try:
        dataset_B_with_meta = read_harmful_prompts_with_metadata(args.harm_file_path_B)
        if not dataset_B_with_meta:
            logger.error(f"Failed to load any valid metadata from {args.harm_file_path_B}")
            return
        logger.info(f"Loaded pool B: {len(dataset_B_with_meta)} items (with metadata)")
    except Exception as e:
        logger.error(f"Failed to load pool B: {e}")
        return

    # --- 3. hardware configuration logic --- 
    try:
        # get device configuration
        inference_devices = config.inference.devices
        judger_devices = config.judger.devices
        
        if not inference_devices or not judger_devices:
            raise ValueError("inference_devices and judger_devices must be specified in the configuration.")
        
        # compute tensor-parallel size from devices string
        inference_tp_size = len(inference_devices.split(','))
        judger_tp_size = len(judger_devices.split(','))
        
        if inference_tp_size == 0 or judger_tp_size == 0:
            raise ValueError("devices string must not be empty.")

        logger.info(f"Inference -> GPUs [{inference_devices}] (TP={inference_tp_size})")
        logger.info(f"Judger -> GPUs [{judger_devices}] (TP={judger_tp_size})")

    except Exception as e:
        logger.error(f"GPU allocation strategy failed: {e}")
        return
    
    # inference model configuration
    inference_config = {
        "model_dir": config.inference.model_dir,
        "adapter_path": getattr(config.inference, 'adapter_path', None),  # supports QLora adapter
        "tensor_parallel_size": inference_tp_size,
        "max_model_len": config.inference.max_model_len,
        "max_output_tokens": config.inference.max_output_tokens,
        "batch_size": config.inference.batch_size,
        "visible_devices": inference_devices,
        "gpu_memory_utilization": getattr(config.inference, 'gpu_memory_utilization', 0.8)
    }
    
    # judger model configuration
    judger_config = {
        "model_dir": config.judger.model_dir,
        "tensor_parallel_size": judger_tp_size,
        "max_model_len": config.judger.max_model_len,
        "max_output_tokens": config.judger.max_output_tokens,
        "batch_size": config.judger.batch_size,
        "visible_devices": judger_devices,
        "gpu_memory_utilization": getattr(config.judger, 'gpu_memory_utilization', 0.9)
    }
    
    # API generator configuration
    api_config = {
        "model": config.api_generator.model,
        "api_key": config.api_generator.api_key,
        "base_url": config.api_generator.base_url,
        "max_concurrency": config.api_generator.max_concurrency,
    }

    # --- 4. initialize dynamic evolution graph (DynamicEvolutionGraph) ---
    logger.info("Initializing dynamic evolution graph...")
    
    try:
        # To be compatible with the parent class (which expects pure text lists),
        # pass text views for harmful_prompts and pool_dataset_B while keeping
        # the original metadata for fixed_dataset_A/pool_dataset_B_with_meta
        # so outputs can include chosen/rejected. Produce temporary lists that
        # may contain None, then filter and cast to list[str].
        _tmp_harm = [get_prompt_from_item(item) for item in dataset_A]
        harmful_prompts_init = [p for p in _tmp_harm if p is not None]
        harmful_prompts_init = cast(list[str], harmful_prompts_init)

        _tmp_pool = [get_prompt_from_item(item) for item in dataset_B_with_meta]
        pool_B_prompts = [p for p in _tmp_pool if p is not None]
        pool_B_prompts = cast(list[str], pool_B_prompts)

        # build a text-only view of fixed set A to satisfy type requirements while keeping original metadata
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
        logger.error(f"Failed to initialize dynamic evolution graph: {e}")
        return

    # --- 5. start evolution ---
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

    # --- 6. cleanup ---
    # simple memory cleanup, see attack_qlora.py
    import gc
    del peg
    gc.collect()

if __name__ == "__main__":
    try:
        asyncio.run(async_main_mixed())
    except Exception:
        logger.exception("Asynchronous main failed")
        sys.exit(1)