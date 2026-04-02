"""Compute similarity matrix for persona texts and produce a heatmap.

This script encodes persona texts using a model tokenizer+encoder,
computes pairwise cosine similarities, and saves a heatmap image.
"""

import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from `model_path`.

    The model path can be a local directory or a Hugging Face model id.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return model, tokenizer

def encode_texts(model, tokenizer, texts, batch_size=8):
    """Encode a list of texts into embeddings.

    Uses the model's last hidden state CLS token (index 0) as the text vector.
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

def calculate_similarity_matrix(embeddings):
    """Return pairwise cosine similarity matrix for `embeddings`."""
    return cosine_similarity(embeddings)

def load_personas(file_path):
    personas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            personas.append({
                'id': data.get('id'),
                'text': data.get('text'),
                'RtA': data.get('RtA'),
                'ASR': data.get('ASR')
            })
    return personas

def main():
    # Configuration: use environment variables to avoid embedding absolute paths
    model_path = os.environ.get('MODEL_PATH', 'models/bge-m3')
    personas_file = os.environ.get('PERSONAS_FILE', 'result/attack1/gen_40_elite_top35_nodes.jsonl')

    output_dir = os.path.dirname(personas_file)

    model, tokenizer = load_model_and_tokenizer(model_path)

    personas = load_personas(personas_file)
    texts = [p['text'] for p in personas]

    print(f"Encoding {len(texts)} texts...")
    embeddings = encode_texts(model, tokenizer, texts)

    print("Computing similarity matrix...")
    similarity_matrix = calculate_similarity_matrix(embeddings)
    
    # Find similar text pairs and collect statistics
    similar_pairs = []
    n = len(personas)

    for i in range(n):
        for j in range(i + 1, n):
            similar_pairs.append({
                'persona_i': i + 1,
                'persona_j': j + 1,
                'similarity': float(similarity_matrix[i][j]),
                'text_i': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                'text_j': texts[j][:100] + "..." if len(texts[j]) > 100 else texts[j]
            })

    high_similarity_pairs = [pair for pair in similar_pairs if pair['similarity'] >= 0.9]
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    print(f"\n=== Similarity Statistics ===")
    print(f"Pairs with similarity >= 0.9: {len(high_similarity_pairs)}")
    if similar_pairs:
        print(f"Proportion of high-similarity pairs: {len(high_similarity_pairs) / len(similar_pairs) * 100:.2f}%")
    
    # Create a visual similarity matrix
    plt.figure(figsize=(14, 12))
    
    # Create heatmap using matplotlib imshow
    im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0.6, vmax=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Similarity', fontsize=12)
    
    # Axis labels
    plt.title('Persona Text Similarity Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Persona ID', fontsize=12)
    plt.ylabel('Persona ID', fontsize=12)
    
    # Set tick labels
    persona_labels = [f"P{i+1}" for i in range(len(personas))]
    plt.xticks(range(len(personas)), persona_labels, rotation=45)
    plt.yticks(range(len(personas)), persona_labels)
    
    # Add grid lines
    plt.grid(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the image to the input file's parent directory
    output_path = os.path.join(output_dir, 'similarity_matrix_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved similarity heatmap to: {output_path}")

    # Display the plot
    plt.show()
    
    # Summary statistics
    print(f"\n=== Summary ===")
    print(f"Total texts: {len(texts)}")
    print(f"Total pairs: {len(similar_pairs)}")
    if len(similar_pairs) > 0:
        mean_sim = np.mean(similarity_matrix[np.triu_indices(n, k=1)])
        max_sim = np.max(similarity_matrix[np.triu_indices(n, k=1)])
        min_sim = np.min(similarity_matrix[np.triu_indices(n, k=1)])
        print(f"Mean similarity: {mean_sim:.4f}")
        print(f"Max similarity: {max_sim:.4f}")
        print(f"Min similarity: {min_sim:.4f}")
        print(f"Pairs with similarity >= 0.9: {len(high_similarity_pairs)} ({len(high_similarity_pairs) / len(similar_pairs) * 100:.2f}%)")

if __name__ == "__main__":
    main()