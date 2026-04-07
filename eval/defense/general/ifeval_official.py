#!/usr/bin/env python3
"""
使用官方IFEval库进行评估
  
python eval/defense/general/ifeval_official.py \
    --input_file result/defense/general/ifeval/ifeval_input_data.jsonl \
    --response_file result/defense/general/ifeval/ifeval_input_data-Qwen2.5-7B-Instruct-qlora.jsonl \
    --output_dir result/defense/general/ifeval \
    --language en \
    --verbose
"""

import json
import argparse
from pathlib import Path

from ifeval.core.evaluation import Evaluator # type: ignore
from ifeval.languages.en.instructions import instruction_registry # type: ignore
from ifeval.utils.io import read_input_examples, read_responses # type: ignore


def print_report(report: dict, title: str = "IFEval Official Evaluation Results"):
    """Print evaluation report"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    
    # Strict evaluation results
    if "eval_results_strict" in report:
        strict = report["eval_results_strict"]
        print(f"\n【Strict Evaluation - Strict Evaluation】")
        print(f"  Prompt Accuracy: {strict.get('prompt_accuracy', 0):.4f} ({strict.get('prompt_accuracy', 0)*100:.2f}%)")
        print(f"  Instruction Accuracy: {strict.get('instruction_accuracy', 0):.4f} ({strict.get('instruction_accuracy', 0)*100:.2f}%)")
        
        if "prompt_following_rate_by_inst_type" in strict:
            print(f"\n  Accuracy by Instruction Type:")
            sorted_types = sorted(strict["prompt_following_rate_by_inst_type"].items(), 
                                key=lambda x: x[1], reverse=True)
            for inst_type, rate in sorted_types:
                print(f"    {inst_type:30s}: {rate:.4f}")
    
    # Loose evaluation results
    if "eval_results_loose" in report:
        loose = report["eval_results_loose"]
        print(f"\n【Loose Evaluation - Loose Evaluation】")
        print(f"  Prompt Accuracy: {loose.get('prompt_accuracy', 0):.4f} ({loose.get('prompt_accuracy', 0)*100:.2f}%)")
        print(f"  Instruction Accuracy: {loose.get('instruction_accuracy', 0):.4f} ({loose.get('instruction_accuracy', 0)*100:.2f}%)")
        
        if "prompt_following_rate_by_inst_type" in loose:
            print(f"\n  Accuracy by Instruction Type:")
            sorted_types = sorted(loose["prompt_following_rate_by_inst_type"].items(), 
                                key=lambda x: x[1], reverse=True)
            for inst_type, rate in sorted_types:
                print(f"    {inst_type:30s}: {rate:.4f}")
    print("="*70)

def save_results(report: dict, all_outputs: list, output_dir: Path, prefix: str):
    """Save evaluation results"""
    # Save report
    report_file = output_dir / f"{prefix}_official_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_file}")
    
    # Save detailed results
    details_file = output_dir / f"{prefix}_official_details.jsonl"
    with open(details_file, 'w', encoding='utf-8') as f:
        for output in all_outputs:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
    print(f"Detailed results saved to: {details_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Use the official IFEval library to evaluate instruction compliance capabilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input data file path (JSONL format)'
    )
    
    parser.add_argument(
        '--response_file',
        type=str,
        required=True,
        help='Response data file path (JSONL format)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory path (default: ./results)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        choices=['en', 'ru'],
        help='Evaluation language (default: en, supported: en, ru)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed evaluation information'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading input data: {args.input_file}")
    input_examples = read_input_examples(args.input_file)
    print(f"  ✓ Loaded {len(input_examples)} input examples")
    
    print(f"\nLoading response data: {args.response_file}")
    responses = read_responses(args.response_file)
    print(f"  ✓ Loaded {len(responses)} responses")
    
    print(f"\nInitializing evaluator...")
    evaluator = Evaluator(instruction_registry)
    
    print(f"\nStarting evaluation...")
    report, all_outputs = evaluator.evaluate(input_examples, responses)
    
    # Print report
    print_report(report)
    
    # Save results
    response_file_name = Path(args.response_file).stem
    save_results(report, all_outputs, output_dir, response_file_name)
    
    print(f"\n✓ Evaluation completed!")
    
    # Print key metrics for comparison
    print(f"\n【Key Metrics】")
    strict_prompt = report["eval_results_strict"]["prompt_accuracy"]
    loose_prompt = report["eval_results_loose"]["prompt_accuracy"]
    strict_inst = report["eval_results_strict"]["instruction_accuracy"]
    loose_inst = report["eval_results_loose"]["instruction_accuracy"]
    
    print(f"Strict  - Prompt: {strict_prompt:.4f}, Instruction: {strict_inst:.4f}")
    print(f"Loose   - Prompt: {loose_prompt:.4f}, Instruction: {loose_inst:.4f}")

if __name__ == "__main__":
    main()
