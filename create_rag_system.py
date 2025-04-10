from llama_cpp import Llama
import tiktoken
from typing import Dict

class RAGSystem:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.llm = Llama(
            model_path="llama-2-7b-chat.Q4_K_M.gguf",  # Your local model
            n_ctx=4096,  # Context window size
            n_threads=6  # CPU threads to use
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def retrieve(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve relevant chunks from vector DB"""
        results = self.vector_db.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0]
        }
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        prompt = f"""Answer the question using ONLY the following context. Cite sources like [Source: filename].

Context:
{context}

Question: {query}
Answer:"""
        
        response = self.llm.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.3,
            stop=["\n\n", "Q:"]
        )
        return response['choices'][0]['text'].strip()
    
    def query(self, question: str) -> Dict:
        """End-to-end RAG query"""
        # Retrieve relevant chunks
        retrieved = self.retrieve(question)
        
        # Format context with sources
        context = "\n\n".join(
            f"{doc}\n[Source: {meta['source']}]" 
            for doc, meta in zip(retrieved['documents'], retrieved['metadatas'])
        )
        
        # Generate answer
        answer = self.generate_response(question, context)
        
        return {
            'answer': answer,
            'sources': list(set(m['source'] for m in retrieved['metadatas']))
        }
