#!/usr/bin/env python3
"""
Deduplicate high-ASR texts by similarity.

Reads a JSONL file of records with `text` and `ASR`, sorts by ASR (desc),
generates TF-IDF embeddings, and greedily selects diverse items by minimizing
maximum similarity to already selected items.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_jsonl_file(file_path: Path):
    data = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse JSON: {e}")
    return data


def sort_by_asr_desc(data):
    return sorted(data, key=lambda x: x.get('ASR', 0), reverse=True)


def generate_embeddings(texts, max_features=512):
    print("Generating TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
    embeddings = vectorizer.fit_transform(texts).toarray()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def calculate_similarity_statistics(embeddings):
    if embeddings.shape[0] < 2:
        return None
    sim = cosine_similarity(embeddings)
    upper = np.triu(sim, k=1)
    vals = upper[upper > 0]
    if vals.size == 0:
        return None
    return {
        'mean_similarity': float(np.mean(vals)),
        'median_similarity': float(np.median(vals)),
        'std_similarity': float(np.std(vals)),
        'min_similarity': float(np.min(vals)),
        'max_similarity': float(np.max(vals)),
        'high_similarity_pairs': int(np.sum(vals > 0.8)),
        'total_pairs': int(vals.size)
    }


def print_similarity_statistics(stats, title, data=None):
    if stats is None:
        print(f"\n{title}: not enough data to compute similarity stats")
        return
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"Number of texts: {len(data) if data is not None else 'unknown'}")
    if data:
        asr_values = [item.get('ASR', 0) for item in data]
        print(f"Mean ASR: {np.mean(asr_values):.4f}")
    print(f"Mean similarity: {stats['mean_similarity']:.4f}")
    print(f"Std similarity: {stats['std_similarity']:.4f}")
    print(f"Min similarity: {stats['min_similarity']:.4f}")
    print(f"Max similarity: {stats['max_similarity']:.4f}")
    print(f"High-similarity pairs (>0.8): {stats['high_similarity_pairs']} ({stats['high_similarity_pairs']/stats['total_pairs']*100:.1f}%)")


def deduplicate_by_similarity(data, embeddings, max_entries=100):
    if not data or embeddings.size == 0:
        return []

    original_stats = calculate_similarity_statistics(embeddings)
    print_similarity_statistics(original_stats, "Similarity stats before deduplication", data)

    unique_data = [data[0]]
    unique_embeddings = [embeddings[0]]

    remaining_data = data[1:]
    remaining_embeddings = embeddings[1:]

    while len(unique_data) < max_entries and remaining_data:
        similarities = cosine_similarity(remaining_embeddings, np.array(unique_embeddings))
        max_similarities = similarities.max(axis=1)
        idx = int(np.argmin(max_similarities))
        unique_data.append(remaining_data[idx])
        unique_embeddings.append(remaining_embeddings[idx])
        remaining_data.pop(idx)
        remaining_embeddings = np.delete(remaining_embeddings, idx, axis=0)

    print(f"\nDeduplication complete, kept {len(unique_data)} unique records")

    unique_embeddings_array = np.array(unique_embeddings)
    dedup_stats = calculate_similarity_statistics(unique_embeddings_array)
    print_similarity_statistics(dedup_stats, "Similarity stats after deduplication", unique_data)

    original_asr = [item.get('ASR', 0) for item in data]
    dedup_asr = [item.get('ASR', 0) for item in unique_data]
    print(f"\n{'='*60}")
    print("Deduplication analysis")
    print(f"{'='*60}")
    print(f"Compression: {len(data)} -> {len(unique_data)} ({len(unique_data)/len(data)*100:.1f}%)")
    if original_stats is not None and dedup_stats is not None:
        print(f"Mean ASR change: {np.mean(original_asr):.4f} -> {np.mean(dedup_asr):.4f} (delta: {np.mean(dedup_asr) - np.mean(original_asr):+.4f})")
        print(f"Mean similarity change: {original_stats['mean_similarity']:.4f} -> {dedup_stats['mean_similarity']:.4f}")
        print(f"High-similarity pairs reduced: {original_stats['high_similarity_pairs']} -> {dedup_stats['high_similarity_pairs']}")

    return unique_data


def save_jsonl_file(data, file_path: Path):
    with file_path.open('w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved {len(data)} records to {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Deduplicate texts by similarity')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, default=None, help='Output JSONL file (default: <input>_dedup.jsonl)')
    parser.add_argument('-m', '--max-entries', type=int, default=100, help='Maximum number of entries to keep')
    parser.add_argument('--max-features', type=int, default=512, help='Max features for TF-IDF vectorizer')
    args = parser.parse_args()

    input_file = args.input
    if not input_file.exists():
        print(f"Error: input file {input_file} not found")
        return

    output_file = args.output or input_file.with_name(input_file.stem + '_dedup.jsonl')

    data = read_jsonl_file(input_file)
    print(f"Read {len(data)} records")
    if not data:
        print("No data read, exiting")
        return

    sorted_data = sort_by_asr_desc(data)
    print(f"Top ASR after sorting: {sorted_data[0].get('ASR', 0)}")
    print(f"Lowest ASR after sorting: {sorted_data[-1].get('ASR', 0)}")

    texts = [item.get('text', '') for item in sorted_data]
    embeddings = generate_embeddings(texts, max_features=args.max_features)
    print(f"Embeddings shape: {embeddings.shape}")

    unique_data = deduplicate_by_similarity(sorted_data, embeddings, max_entries=args.max_entries)
    save_jsonl_file(unique_data, output_file)


if __name__ == '__main__':
    main()
