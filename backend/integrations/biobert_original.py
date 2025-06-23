"""
Original BioBERT Embedding Implementation
Compatible with Pinecone vectors created using the original BioBERT model
"""

import numpy as np
import logging
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)

class OriginalBioBERT:
    """
    Implementation that matches the original BioBERT embedding generation
    Used to create embeddings compatible with existing Pinecone vectors
    """
    
    def __init__(self, model_path: str = "dmis-lab/biobert-v1.1"):
        """
        Initialize with the original BioBERT model
        Args:
            model_path: Path to BioBERT model (default: dmis-lab/biobert-v1.1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load the exact BioBERT model used for Pinecone
            logger.info(f"Loading original BioBERT model: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Original BioBERT loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BioBERT model: {e}")
            raise
    
    def word_vector(self, text: str) -> List[np.ndarray]:
        """
        Get word-level embeddings (768-dimensional for each token)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get all token embeddings
            token_embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
        
        # Convert to numpy and return list of arrays
        word_embeddings = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Skip [CLS] and [SEP] tokens
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                word_embeddings.append(token_embeddings[i].cpu().numpy())
        
        return word_embeddings
    
    def sentence_vector(self, text: str) -> np.ndarray:
        """
        Get sentence-level embedding (768-dimensional)
        Uses the [CLS] token representation as the sentence embedding
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token (first token) as sentence representation
            sentence_embedding = outputs.last_hidden_state[0][0]  # [batch_size=1, seq_len, hidden_size]
        
        # Convert to numpy
        embedding = sentence_embedding.cpu().numpy()
        
        # Normalize (important for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    @property
    def tokens(self) -> List[str]:
        """
        Get the tokens from the last processed text
        (For compatibility with biobert-embedding package)
        """
        if hasattr(self, '_last_tokens'):
            return self._last_tokens
        return []
    
    def process_text(self, text: str) -> dict:
        """
        Process text and return both word and sentence embeddings
        """
        # Store tokens for property access
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        self._last_tokens = [
            token for token in self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            if token not in ['[CLS]', '[SEP]', '[PAD]']
        ]
        
        return {
            'tokens': self._last_tokens,
            'word_embeddings': self.word_vector(text),
            'sentence_embedding': self.sentence_vector(text)
        }

# Wrapper class that matches the biobert-embedding API
class BiobertEmbedding:
    """
    Drop-in replacement for biobert-embedding package
    Uses the original BioBERT model for compatible embeddings
    """
    
    def __init__(self, model_path: str = None):
        # Use the original BioBERT v1.1 model
        if model_path is None:
            model_path = "dmis-lab/biobert-v1.1"
        
        self._biobert = OriginalBioBERT(model_path)
        self._last_tokens = []
    
    def word_vector(self, text: str) -> List[np.ndarray]:
        """Get word embeddings"""
        result = self._biobert.process_text(text)
        self._last_tokens = result['tokens']
        return result['word_embeddings']
    
    def sentence_vector(self, text: str) -> np.ndarray:
        """Get sentence embedding"""
        return self._biobert.sentence_vector(text)
    
    @property
    def tokens(self) -> List[str]:
        """Get tokens from last processed text"""
        return self._last_tokens