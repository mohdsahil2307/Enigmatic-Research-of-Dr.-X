import chromadb
from chromadb.utils import embedding_functions
import numpy as np

def load_embeddings(input_file="embeddings.npz"):
    """Load saved embeddings"""
    data = np.load(input_file, allow_pickle=True)
    return [{
        'id': id_,
        'text': text,
        'embedding': embedding,
        'metadata': metadata
    } for id_, text, embedding, metadata in zip(
        data['ids'],
        data['texts'],
        data['embeddings'],
        data['metadatas']
    )]
    
def create_vector_db(embedded_chunks):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    collection = client.get_or_create_collection(
        name="dr_x_research"
    )
    # Clean metadata for ChromaDB compatibility
    def clean_metadata(metadata):
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
    
    # Batch insertion
    batch_size = 100
    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i:i+batch_size]
        
        # If using pre-computed embeddings
        collection.add(
            ids=[chunk['id'] for chunk in batch],
            documents=[chunk['text'] for chunk in batch],
            metadatas=[clean_metadata(chunk['metadata']) for chunk in batch],
            embeddings=[chunk['embedding'] for chunk in batch]
        )
    return collection
    
vector_db = create_vector_db(load_embeddings())

