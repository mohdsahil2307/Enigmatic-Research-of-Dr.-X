import os
import json
from nomic import embed
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict
from nomic import embed
import numpy as np

def enhance_chunks(chunks: List[Dict]) -> List[Dict]:
    """Adds RAG-specific metadata and cleans text"""
    enhanced = []
    for chunk in chunks:
        # Clean text and detect content type
        clean_text = ' '.join(chunk['text'].split())
        is_table = chunk['type'] == 'table'
        has_citation = 'doi.org/' in clean_text.lower() or 'http' in clean_text.lower()
        
        # Ensure metadata is never None
        metadata = chunk.get('metadata', {})
        if metadata is None:
            metadata = {}

        enhanced.append({
            'id': chunk['chunk_id'],
            'text': clean_text,
            'metadata': {
                'source': metadata.get('source', 'unknown'),
                'content_type': 'table' if is_table else 'text',
                'page': int(chunk['chunk_id'].split('_page')[-1].split('_')[0]) if '_page' in chunk['chunk_id'] else None,
                'tokens': chunk.get('tokens', 0),
                'has_citation': has_citation,
                'is_heading': chunk.get('tokens', 0) < 10 and clean_text.isupper()
            }
        })
    return enhanced

# Usage
with open('extracted_text.json', encoding='utf-8') as f:
    raw_chunks = json.load(f)
processed_chunks = enhance_chunks(raw_chunks)

def generate_embeddings(chunks: List[Dict], batch_size=100) -> List[Dict]:
    """Generates embeddings in batches to avoid timeout"""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk['text'] for chunk in batch]
        
        try:
            embeddings = embed.text(
                texts=texts,
                model='nomic-embed-text-v1.5',
                task_type='search_document'
            )['embeddings']
            
            for j, chunk in enumerate(batch):
                chunk['embedding'] = embeddings[j]
        except Exception as e:
            print(f"Failed batch {i//batch_size}: {str(e)}")
    
    return chunks

def save_embeddings(embedded_chunks, output_file="embeddings.npz"):
    """Save embeddings in compact numpy format"""
    np.savez_compressed(
        output_file,
        ids=[chunk['id'] for chunk in embedded_chunks],
        texts=[chunk['text'] for chunk in embedded_chunks],
        embeddings=np.array([chunk['embedding'] for chunk in embedded_chunks]),
        metadatas=[chunk['metadata'] for chunk in embedded_chunks]
    )

embedded_chunks = generate_embeddings(processed_chunks)
save_embeddings(embedded_chunks)