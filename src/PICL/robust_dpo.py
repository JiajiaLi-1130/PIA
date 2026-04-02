# Imports
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference, DPOTrainer


@dataclass
class RobustDPOConfig(DPOConfig):
    """Robust DPO configuration extending DPOConfig.

    Adds optional SFT mixed loss and PIC regularization parameters.
    """
    sft_alpha: float = field(default=0.0, metadata={"help": "Coefficient for SFT loss (balances DPO and SFT)"})
    pic_lambda: float = field(default=0.0, metadata={"help": "Coefficient for PIC regularization; >0 enables PIC"})
    pic_top_k: int = field(default=3, metadata={"help": "Top-K tokens considered when computing PIC loss per token"})

class RobustDataCollator(DataCollatorForPreference):
    """Custom data collator that supports persona masks and optional clean prompts for PIC."""
    def __call__(self, features):
        batch = super().__call__(features)

        if "persona_attention_mask" in features[0]:
            p_masks = []
            for f in features:
                mask = f["persona_attention_mask"]
                p_masks.append(mask if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
                               else torch.tensor(mask, dtype=torch.bool))
            batch["persona_attention_mask"] = self._pad_left(p_masks, pad_value=False)

        if "is_dpo" in features[0]:
            batch["is_dpo"] = torch.tensor([f["is_dpo"] for f in features], dtype=torch.bool)

        if "has_persona" in features[0]:
            batch["has_persona"] = torch.tensor([f["has_persona"] for f in features], dtype=torch.bool)

        # Handle optional clean_prompt_input_ids only when present and non-empty
        if "clean_prompt_input_ids" in features[0]:
            has_valid_clean = any(f.get("clean_prompt_input_ids") and len(f.get("clean_prompt_input_ids", [])) > 0 for f in features)
            if has_valid_clean:
                clean_prompts, clean_masks = [], []
                for f in features:
                    cp = f.get("clean_prompt_input_ids", [])
                    if cp and isinstance(cp, list) and len(cp) > 0:
                        cp_tensor = torch.tensor(cp, dtype=torch.long)
                    else:
                        cp_tensor = torch.tensor([], dtype=torch.long)
                    clean_prompts.append(cp_tensor)
                    clean_masks.append(torch.ones_like(cp_tensor, dtype=torch.bool))

                batch["clean_prompt_input_ids"] = self._pad_left(clean_prompts, pad_value=self.pad_token_id)
                batch["clean_prompt_attention_mask"] = self._pad_left(clean_masks, pad_value=False)

        return batch

    def _pad_left(self, tensors, pad_value=0):
        """Left-pad a list of 1D tensors to the same length and stack them."""
        if not tensors:
            return torch.tensor([], dtype=torch.long)

        tensors = [t for t in tensors if t is not None]
        if not tensors:
            return torch.tensor([], dtype=torch.long)

        lengths = [len(t) for t in tensors]
        max_len = max(lengths) if lengths else 0

        if max_len == 0:
            dtype = tensors[0].dtype if tensors else torch.long
            return torch.zeros((len(tensors), 0), dtype=dtype)

        padded_batch = []
        for t in tensors:
            pad_len = max_len - len(t)
            if pad_len > 0:
                if t.dtype == torch.bool:
                    pad_tensor = torch.full((pad_len,), pad_value, dtype=torch.bool, device=t.device)
                else:
                    pad_tensor = torch.full((pad_len,), float(pad_value), dtype=t.dtype, device=t.device)
                padded = torch.cat([pad_tensor, t])
            else:
                padded = t
            padded_batch.append(padded)
        return torch.stack(padded_batch)

class RobustDPOTrainer(DPOTrainer):
    """Trainer that integrates SFT mixed loss and PIC (persona invariance consistency) regularization."""
    args: RobustDPOConfig

    def __init__(self, *args_positional, args=None, **kwargs):
        if args is not None and not isinstance(args, RobustDPOConfig):
            raise TypeError("RobustDPOTrainer requires RobustDPOConfig.")
        super().__init__(*args_positional, args=args, **kwargs)
        self._static_graph_set = False

    def _setup_ddp_static_graph(self):
        """Compatibility fix for DDP + LoRA + gradient checkpointing.

        Ensures static graph mode is set when needed to avoid DDP issues.
        """
        if self._static_graph_set:
            return
        if hasattr(self, 'accelerator') and self.accelerator.num_processes > 1:
            if self.args.gradient_checkpointing and hasattr(self.model, '_set_static_graph'):
                try:
                    self.model._set_static_graph()  # type: ignore
                    self._static_graph_set = True
                except Exception:
                    pass

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self._static_graph_set:
            self._setup_ddp_static_graph()
        return super().training_step(model, inputs, num_items_in_batch)
    
    def _get_top_k_mask(self, logits, k):
        """Return a mask selecting the top-k values per row in logits."""
        vocab_size = logits.shape[-1]
        k = min(k, vocab_size)
        if k <= 0:
            return torch.zeros_like(logits, dtype=torch.float)
        values, _ = torch.topk(logits, k, dim=-1)
        min_top_k = values[:, -1:]
        mask = (logits >= min_top_k).float()
        return mask

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], is_ref_model: bool = False,
        need_sft_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Standard DPO forward. Also computes per-sample NLL for optional SFT loss."""
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)
        input_ids = torch.cat((concatenated_batch["prompt_input_ids"], concatenated_batch["completion_input_ids"]), dim=1)
        attention_mask = torch.cat((concatenated_batch["prompt_attention_mask"], concatenated_batch["completion_attention_mask"]), dim=1)

        model_kwargs = {"use_cache": False, "attention_mask": attention_mask}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
            
        outputs = model(input_ids=input_ids, **model_kwargs)
        logits = outputs.logits

        # Compute token log-probabilities efficiently (reduce roll ops and memory)
        labels = input_ids[:, 1:].contiguous()  # slice directly without roll
        loss_mask = torch.cat((torch.zeros_like(concatenated_batch["prompt_attention_mask"]), concatenated_batch["completion_attention_mask"]), dim=1)
        loss_mask = loss_mask[:, 1:].bool()
        
        logits_for_labels = logits[:, :-1, :].contiguous()

        # Performance: compute log_softmax once and gather to avoid Python-level loops
        log_probs = logits_for_labels.log_softmax(-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2).float()
        
        per_token_logps = per_token_logps * loss_mask.float()
        all_logps = per_token_logps.sum(-1)
        
        len_chosen = batch["chosen_input_ids"].shape[0] # type: ignore
        output = {
            "chosen_logps": all_logps[:len_chosen],
            "rejected_logps": all_logps[len_chosen:],
            "logits": logits,
            "prompt_attention_mask": concatenated_batch["prompt_attention_mask"],
            "completion_attention_mask": concatenated_batch["completion_attention_mask"]
        }
        
        # Compute SFT loss (NLL) only when requested; reuse existing tensors
        if need_sft_loss:
            chosen_per_token_logps = per_token_logps[:len_chosen]
            chosen_loss_mask_for_nll = loss_mask[:len_chosen]
            masked_nll = -chosen_per_token_logps * chosen_loss_mask_for_nll.float()
            # Handle zero-token samples explicitly: their loss should be 0
            token_counts = chosen_loss_mask_for_nll.float().sum(dim=1)
            masked_nll_sum = masked_nll.sum(dim=1)
            per_sample_nll = torch.zeros_like(masked_nll_sum)
            nonzero_mask = token_counts > 0
            per_sample_nll[nonzero_mask] = masked_nll_sum[nonzero_mask] / token_counts[nonzero_mask]
            output["per_sample_nll"] = per_sample_nll

        return output

    def compute_pic_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        need_pic_mask: torch.Tensor,
        camo_logits: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        chosen_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Compute PIC (persona invariance consistency) loss.

        The objective is to pull persona-conditioned outputs closer to
        persona-free outputs. Loss is based on KL(Clean || Camouflaged).

        For distributed stability this keeps the graph connected even when
        there are no PIC samples in the batch.
        """
        device = need_pic_mask.device
        
        # Check whether there are PIC samples and required clean prompt data
        has_pic_samples = need_pic_mask.any().item()
        has_clean_prompt_data = "clean_prompt_input_ids" in batch and batch["clean_prompt_input_ids"].shape[1] > 0

        # If no PIC samples or missing data, return a zero (connected) loss.
        # camo_logits must be provided by caller to keep the graph connected.
        assert camo_logits is not None, "camo_logits must be provided"
        
        if not has_pic_samples or not has_clean_prompt_data:
            dummy_loss = camo_logits.sum() * 0.0
            return dummy_loss, 0, torch.tensor([], device=device)
        
        # Build clean context batch (clean prompt + chosen response)
        clean_prompt_ids = batch["clean_prompt_input_ids"]
        clean_prompt_mask = batch["clean_prompt_attention_mask"]
        chosen_ids = batch["chosen_input_ids"]

        # Use provided masks or fall back to masks from batch
        if chosen_mask is None:
            chosen_mask = batch["chosen_attention_mask"]
        if prompt_mask is None:
            prompt_mask = batch["prompt_attention_mask"]

        # Ensure masks are present
        assert chosen_mask is not None and prompt_mask is not None, "chosen_mask and prompt_mask must not be None"
        # Fixed response length (aligned with camo_logits)
        response_len_fixed = chosen_mask.shape[1]

        # Select indices for PIC computation
        pic_indices = need_pic_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(pic_indices) == 0:
            # Even if there are no PIC samples, return a zero loss that
            # remains connected to the computation graph to preserve stability.
            dummy_loss = camo_logits.sum() * 0.0
            return dummy_loss, 0, torch.tensor([], device=device)

        # Perform clean forward only for PIC samples to save memory/compute
        clean_prompt_ids_pic = clean_prompt_ids[pic_indices]
        clean_prompt_mask_pic = clean_prompt_mask[pic_indices]
        
        # Use chosen response tensors produced by concatenated_forward
        response_len_fixed = chosen_mask.shape[1]
        
        # Build inputs for clean forward with matching response lengths
        chosen_ids_subset = chosen_ids[pic_indices]  # [n_pic, response_len_fixed]
        chosen_mask_subset = chosen_mask[pic_indices]  # [n_pic, response_len_fixed]
        
        resp_ids_tensor = chosen_ids_subset
        resp_mask_tensor = chosen_mask_subset

        clean_input_ids = torch.cat((clean_prompt_ids_pic, resp_ids_tensor), dim=1)
        clean_attention_mask = torch.cat((clean_prompt_mask_pic, resp_mask_tensor), dim=1)
        
        # Compute clean (target) logits under no-grad and unwrap model
        # to avoid DDP synchronization during this inference step
        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(model)
            clean_outputs = unwrapped_model(input_ids=clean_input_ids, attention_mask=clean_attention_mask, use_cache=False)
            clean_logits_pic = clean_outputs.logits.detach()

        # Use supplied camouflaged logits (from concatenated_forward)
        
        kl_losses = []
        top_k = self.args.pic_top_k

        # Use masks from concatenated_forward to determine true prompt/response lengths
        prompt_len = prompt_mask.shape[1]
        response_len = chosen_mask.shape[1]

        # Clean prompt length comes from the batch
        clean_prompt_len = batch["clean_prompt_input_ids"].shape[1]

        # Iterate only over samples that require PIC; clean_logits was computed
        # for the pic_indices subset.
        for local_idx, global_idx in enumerate(pic_indices):
            i = int(global_idx.item())
            
            # Compute valid response length (exclude padding)
            valid_len = int(chosen_mask_subset[local_idx].sum().item())
            if valid_len == 0:
                continue
            
            # Compute start indices for response segments in logits
            camo_start_idx = prompt_len - 1
            clean_start_idx = clean_prompt_len - 1
            
            # Validate slice endpoints are in range
            if (camo_start_idx < 0 or camo_start_idx + response_len > camo_logits.shape[1] or
                clean_start_idx < 0 or clean_start_idx + response_len > clean_logits_pic.shape[1]):
                continue
            
            # Extract Clean and Camouflaged logits
            tgt_logits = clean_logits_pic[local_idx, clean_start_idx : clean_start_idx + response_len, :]
            src_logits = camo_logits[i, camo_start_idx : camo_start_idx + response_len, :]
            
            # Verify extracted logits sizes match expected lengths
            if tgt_logits.shape[0] != response_len or src_logits.shape[0] != response_len or tgt_logits.shape != src_logits.shape:
                continue
            
            # Compute KL only on valid (non-padded) positions
            response_mask = chosen_mask_subset[local_idx, :response_len]
            valid_indices = response_mask.nonzero(as_tuple=False).squeeze(-1)
            
            # Defensive: ensure there are valid indices
            if valid_indices.numel() == 0:
                continue
            
            # Extract logits for valid positions only
            tgt_logits_valid = tgt_logits[valid_indices]
            src_logits_valid = src_logits[valid_indices]
            
            # Defensive: skip if extracted logits are empty
            if tgt_logits_valid.shape[0] == 0:
                continue
            
            # Dynamically adjust top_k to avoid out-of-bounds
            vocab_size = tgt_logits_valid.shape[-1]
            if vocab_size <= 0 or vocab_size > 200000:
                continue
            
            actual_top_k = min(self.args.pic_top_k, vocab_size)
            if actual_top_k <= 0:
                continue
            
            # Create top-k mask for valid positions
            tgt_top_k_mask = self._get_top_k_mask(tgt_logits_valid, actual_top_k)
            
            # Compute distributions
            src_log_probs = F.log_softmax(src_logits_valid, dim=-1)
            tgt_probs = F.softmax(tgt_logits_valid, dim=-1)
            
            # Defensive: skip if numerical issues
            if torch.isnan(src_log_probs).any() or torch.isnan(tgt_probs).any():
                continue

            # Compute KL divergence per token
            kl_per_token = F.kl_div(src_log_probs, tgt_probs, reduction='none')
            kl_per_token_masked = kl_per_token * tgt_top_k_mask
            kl = kl_per_token_masked.sum() / len(valid_indices)
            
            # Defensive: sanity-check KL value
            if torch.isnan(kl) or torch.isinf(kl) or kl > 100.0:
                continue
            kl_losses.append(kl)

        if not kl_losses:
            return torch.tensor(0.0, device=need_pic_mask.device), 0, torch.tensor([], device=device)
        
        per_sample = torch.stack(kl_losses)
        avg_loss = per_sample.mean()
        return avg_loss, len(kl_losses), per_sample

    def get_batch_loss_metrics(
        self, model, batch, train_eval="train"
    ):
        """Compute loss and metrics integrating DPO, optional SFT, and PIC.

        Per-sample losses are averaged to ensure correct gradient accumulation.
        """
        metrics = {}
        chosen_ids = batch["chosen_input_ids"]
        len_chosen = len(chosen_ids) if isinstance(chosen_ids, list) else chosen_ids.shape[0]
        
        sft_alpha = self.args.sft_alpha
        pic_lambda = self.args.pic_lambda
        
        # 1. Policy forward pass — standard DPO. Run once to get device.
        if "is_dpo" in batch:
            is_dpo_raw = batch["is_dpo"]
            # If it's a list, convert to tensor for checks
            is_dpo_for_check = is_dpo_raw if isinstance(is_dpo_raw, torch.Tensor) else torch.tensor(is_dpo_raw, dtype=torch.bool)
            has_dpo = is_dpo_for_check.any().item()
            has_sft = (~is_dpo_for_check).any().item()
        else:
            has_dpo = True
            has_sft = False
        
        # Compute SFT loss only if SFT samples exist and SFT is enabled
        need_sft_loss = bool(sft_alpha > 0 and train_eval == "train" and has_sft)
        need_pic_loss = bool(pic_lambda > 0 and train_eval == "train")

        policy_output = self.concatenated_forward(model, batch, is_ref_model=False, need_sft_loss=need_sft_loss)
        device = policy_output["chosen_logps"].device

        # Move is_dpo to the correct device
        if "is_dpo" in batch:
            is_dpo = is_dpo_raw.to(device) if isinstance(is_dpo_raw, torch.Tensor) else torch.tensor(is_dpo_raw, dtype=torch.bool, device=device)
        else:
            is_dpo = torch.ones(len_chosen, dtype=torch.bool, device=device)

        # 2. Reference forward pass — compute only if DPO samples are present
        if has_dpo:
            with torch.no_grad():
                if "ref_chosen_logps" in batch:
                    ref_chosen_logps_raw = batch["ref_chosen_logps"]
                    ref_rejected_logps_raw = batch["ref_rejected_logps"]
                    ref_chosen_logps = ref_chosen_logps_raw.to(device) if isinstance(ref_chosen_logps_raw, torch.Tensor) else torch.tensor(ref_chosen_logps_raw, device=device)
                    ref_rejected_logps = ref_rejected_logps_raw.to(device) if isinstance(ref_rejected_logps_raw, torch.Tensor) else torch.tensor(ref_rejected_logps_raw, device=device)
                else:
                    ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)  # type: ignore
        else:
            # Use dummy zeros when no DPO samples are present
            ref_chosen_logps = torch.zeros_like(policy_output["chosen_logps"])
            ref_rejected_logps = torch.zeros_like(policy_output["rejected_logps"])

        # 3. Calculate losses: build per-sample losses and average
        loss_type = self.args.loss_type[0] if isinstance(self.args.loss_type, list) else self.args.loss_type
        
        policy_chosen = policy_output["chosen_logps"]
        policy_rejected = policy_output["rejected_logps"]
        
        # Compute DPO loss (per-sample)
        dpo_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen_logps, ref_rejected_logps, loss_type=loss_type  # type: ignore
        )
        
        # Initialize per-sample losses (following TRL practice)
        per_sample_losses = torch.zeros(len_chosen, dtype=dpo_losses.dtype, device=device)
        
        # A. Fill in DPO samples' losses
        n_dpo = 0
        per_sample_dpo = None
        if is_dpo.any():
            # Assign DPO losses only for DPO examples and count them
            per_sample_dpo = dpo_losses[is_dpo]
            per_sample_losses[is_dpo] = per_sample_dpo
            n_dpo = int(per_sample_dpo.numel())
        
        # B. Fill SFT samples' loss (replace, not add)
        n_sft = 0
        sft_per_sample = None
        if need_sft_loss and "per_sample_nll" in policy_output:
            is_sft = ~is_dpo
            if is_sft.any():
                # Compute SFT per-sample NLL only for SFT examples
                sft_per_sample = sft_alpha * policy_output["per_sample_nll"][is_sft]
                per_sample_losses[is_sft] = sft_per_sample
                n_sft = int(sft_per_sample.numel())
        
        # Average per-sample losses
        main_loss = per_sample_losses.mean()
        total_loss = main_loss
        
        # C. PIC Loss - add as regularization (similar to TRL aux_loss)
        # For distributed stability always call compute_pic_loss; it returns
        # a connected zero-loss when no PIC samples exist.
        n_pic = 0
        per_sample_pic_losses = torch.tensor([], device=device)  # initialize empty tensor
        if need_pic_loss:
            has_persona_batch = batch.get("has_persona")
            if has_persona_batch is not None:
                has_persona_tensor = has_persona_batch.to(device) if isinstance(has_persona_batch, torch.Tensor) else torch.tensor(has_persona_batch, dtype=torch.bool, device=device)
                need_pic_mask = is_dpo & has_persona_tensor
            else:
                need_pic_mask = is_dpo
            
            # Always call compute_pic_loss to keep graphs consistent across GPUs.
            # Reuse logits from policy_output when available.
            policy_logits = policy_output.get("logits")
            if policy_logits is not None:
                # policy_logits shape: [batch_size*2, seq_len, vocab_size] (chosen + rejected)
                # Extract the chosen portion
                chosen_logits = policy_logits[:len_chosen]
                # Extract corresponding masks (concatenated_batch returns tensors)
                prompt_attn_mask = policy_output["prompt_attention_mask"][:len_chosen]
                chosen_attn_mask = policy_output["completion_attention_mask"][:len_chosen]
            else:
                chosen_logits = None
                prompt_attn_mask = None
                chosen_attn_mask = None
            
            pic_loss, n_pic, per_sample_pic_losses = self.compute_pic_loss(
                model, batch, need_pic_mask, 
                camo_logits=chosen_logits,
                prompt_mask=prompt_attn_mask,
                chosen_mask=chosen_attn_mask
            )
            if n_pic > 0:
                total_loss = total_loss + pic_lambda * pic_loss
        
        # Note: do not log total_loss here because Trainer logs 'loss'
        # which corresponds to total_loss, avoiding duplicate reporting.

        # Metrics
        if has_dpo:
            dpo_indices = is_dpo.nonzero(as_tuple=False).squeeze(-1)
            metrics["rewards/chosen"] = float(chosen_rewards[dpo_indices].mean().detach().item())
            metrics["rewards/rejected"] = float(rejected_rewards[dpo_indices].mean().detach().item())
            metrics["rewards/accuracies"] = float((chosen_rewards[dpo_indices] > rejected_rewards[dpo_indices]).float().mean().detach().item())
            metrics["rewards/margins"] = float((chosen_rewards[dpo_indices] - rejected_rewards[dpo_indices]).mean().detach().item())
        
        # Record raw PIC KL divergence (unscaled) for analysis
        if n_pic > 0 and per_sample_pic_losses.numel() > 0:
            # per_sample_pic_losses stores raw KL values
            metrics["metrics/pic_kl"] = float(per_sample_pic_losses.mean().detach().item())

        # --- Global aggregation (optional, for distributed setups) ---
        try:
            dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
        except Exception:
            dist_available = False

        # Prepare local tensors for sums and counts
        # DPO local sum and count (from per_sample_dpo)
        if per_sample_dpo is not None and n_dpo > 0:
            local_dpo_sum = per_sample_dpo.sum()
            local_dpo_count = torch.tensor(n_dpo, dtype=torch.long, device=device)
        else:
            local_dpo_sum = torch.tensor(0.0, device=device)
            local_dpo_count = torch.tensor(0, dtype=torch.long, device=device)
        # SFT local sum and count (from sft_per_sample)
        if sft_per_sample is not None and n_sft > 0:
            local_sft_sum = sft_per_sample.sum()
            local_sft_count = torch.tensor(n_sft, dtype=torch.long, device=device)
        else:
            local_sft_sum = torch.tensor(0.0, device=device)
            local_sft_count = torch.tensor(0, dtype=torch.long, device=device)
        # PIC local sum and count (from per_sample_pic_losses)
        if n_pic > 0 and per_sample_pic_losses.numel() > 0:
            # per_sample_pic_losses contains raw KL values; metrics show scaled average
            local_pic_sum = per_sample_pic_losses.sum() * pic_lambda
            local_pic_count = torch.tensor(n_pic, dtype=torch.long, device=device)
        else:
            local_pic_sum = torch.tensor(0.0, device=device)
            local_pic_count = torch.tensor(0, dtype=torch.long, device=device)

        if dist_available and torch.distributed.get_world_size() > 1:
            # Use all-reduce to aggregate sums (float) and counts (long)
            # sums use float tensors
            global_dpo_sum = local_dpo_sum.clone()
            torch.distributed.all_reduce(global_dpo_sum, op=torch.distributed.ReduceOp.SUM)
            global_sft_sum = local_sft_sum.clone()
            torch.distributed.all_reduce(global_sft_sum, op=torch.distributed.ReduceOp.SUM)
            global_pic_sum = local_pic_sum.clone()
            torch.distributed.all_reduce(global_pic_sum, op=torch.distributed.ReduceOp.SUM)
            # counts use long tensors (float would also work, but keep integer semantics)
            global_dpo_count = local_dpo_count.clone()
            torch.distributed.all_reduce(global_dpo_count, op=torch.distributed.ReduceOp.SUM)
            global_sft_count = local_sft_count.clone()
            torch.distributed.all_reduce(global_sft_count, op=torch.distributed.ReduceOp.SUM)
            global_pic_count = local_pic_count.clone()
            torch.distributed.all_reduce(global_pic_count, op=torch.distributed.ReduceOp.SUM)
            # Compute global means (only when global counts > 0)
            gd = int(global_dpo_count.item())
            gs = int(global_sft_count.item())
            gp = int(global_pic_count.item())
            # Record float metrics to avoid dtype issues in TRL logging
            metrics["losses/dpo"] = float((global_dpo_sum / gd).detach().item()) if gd > 0 else 0.0
            metrics["losses/sft"] = float((global_sft_sum / gs).detach().item()) if gs > 0 else 0.0
            metrics["losses/pic"] = float((global_pic_sum / gp).detach().item()) if gp > 0 else 0.0
        else:
            # Non-distributed / single-card: keep local counts and means
            metrics["losses/dpo"] = float((local_dpo_sum / local_dpo_count).item()) if local_dpo_count.item() > 0 else 0.0
            metrics["losses/sft"] = float((local_sft_sum / local_sft_count).item()) if local_sft_count.item() > 0 else 0.0
            metrics["losses/pic"] = float((local_pic_sum / local_pic_count).item()) if local_pic_count.item() > 0 else 0.0

        # Add step info (instead of epoch)
        if hasattr(self, 'state') and self.state is not None:
            metrics["step"] = float(self.state.global_step)
        # Ensure all metrics are floats to avoid dtype issues in TRL logging
        metrics = {k: float(v) for k, v in metrics.items()}

        return total_loss, metrics
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Override log to remove the `epoch` field when present.

        Accepts positional and keyword args for compatibility across
        different Transformers/TRL versions.
        """
        # If logs is not a dict, forward to the parent implementation
        try:
            if isinstance(logs, dict) and "epoch" in logs:
                logs.pop("epoch")
        except Exception:
            pass

        super().log(logs, *args, **kwargs)