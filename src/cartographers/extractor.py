import numpy as np
from typing import List, Dict, Union
from pathlib import Path


class EmbeddingExtractor:
    """
    Model-agnostic embedding extractor supporting both BERT and Ollama models.
    """
    
    def __init__(self, model_type: str, model_name: str):
        """
        Initialize the extractor with specified model type.
        
        Args:
            model_type: Either 'bert' or 'ollama'
            model_name: Model identifier (e.g., 'bert-base-uncased' or 'llama2')
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        if self.model_type == 'bert':
            self._init_bert()
        elif self.model_type == 'ollama':
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_bert(self):
        """Initialize BERT model and tokenizer."""
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.model = ollama
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
    
    def _extract_bert_embedding(self, text: str) -> np.ndarray:
        """Extract embedding using BERT model."""
        import torch
        
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding or mean pooling
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return embedding
    
    def _extract_ollama_embedding(self, text: str) -> np.ndarray:
        """Extract embedding using Ollama model."""
        response = self.model.embeddings(model=self.model_name, prompt=text)
        return np.array(response['embedding'])
    
    def extract_embedding(self, text: str) -> np.ndarray:
        """
        Extract embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding as numpy array
        """
        if self.model_type == 'bert':
            return self._extract_bert_embedding(text)
        elif self.model_type == 'ollama':
            return self._extract_ollama_embedding(text)
    
    def load_concepts(self, filepath: str = 'concepts_list.txt') -> List[str]:
        """
        Load concepts from a text file.
        
        Args:
            filepath: Path to the concepts file
            
        Returns:
            List of concepts
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Concepts file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            concepts = [line.strip() for line in f if line.strip()]
        
        return concepts
    
    def extract_all_embeddings(self, concepts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for all concepts.
        
        Args:
            concepts: List of concept strings
            
        Returns:
            Dictionary mapping concepts to their embeddings
        """
        embeddings = {}
        
        for concept in concepts:
            try:
                embedding = self.extract_embedding(concept)
                embeddings[concept] = embedding
                print(f"✓ Extracted embedding for: {concept}")
            except Exception as e:
                print(f"✗ Failed to extract embedding for '{concept}': {e}")
        
        return embeddings
    
    def extract_from_file(self, filepath: str = 'concepts_list.txt') -> Dict[str, np.ndarray]:
        """
        Load concepts from file and extract all embeddings.
        
        Args:
            filepath: Path to the concepts file
            
        Returns:
            Dictionary mapping concepts to their embeddings
        """
        concepts = self.load_concepts(filepath)
        return self.extract_all_embeddings(concepts)


# Usage Examples
if __name__ == "__main__":
    # Example 1: Using BERT
    print("=" * 50)
    print("BERT Example")
    print("=" * 50)
    
    bert_extractor = EmbeddingExtractor(
        model_type='bert',
        model_name='bert-base-uncased'
    )
    
    # Extract from file
    bert_embeddings = bert_extractor.extract_from_file('concepts_list.txt')
    print(f"\nExtracted {len(bert_embeddings)} BERT embeddings")
    
    # Example 2: Using Ollama
    print("\n" + "=" * 50)
    print("Ollama Example")
    print("=" * 50)
    
    ollama_extractor = EmbeddingExtractor(
        model_type='ollama',
        model_name='llama2'  # or 'mistral', 'nomic-embed-text', etc.
    )
    
    # Extract from file
    ollama_embeddings = ollama_extractor.extract_from_file('concepts_list.txt')
    print(f"\nExtracted {len(ollama_embeddings)} Ollama embeddings")
    
    # Save embeddings (optional)
    np.savez('embeddings.npz', **bert_embeddings)