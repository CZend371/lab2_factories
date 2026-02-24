from typing import Dict, Any
from .base import BaseFeatureGenerator
from app.dataclasses import Email
from sentence_transformers import SentenceTransformer
import numpy as np

class SpamFeatureGenerator(BaseFeatureGenerator):
    """Generates spam detection features from email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}".lower()
        
        # Spam word detection
        spam_words = ['free', 'winner', 'congratulations', 'click here', 'limited time', 
                      'act now', 'urgent', 'special offer', 'guaranteed', 'no risk',
                      'cash', 'money back', 'amazing', 'incredible', 'unbeatable']
        
        has_spam_words = int(any(word in all_text for word in spam_words))
        
        return {"has_spam_words": has_spam_words}
    
    @property
    def feature_names(self) -> list[str]:
        return ["has_spam_words"]


class AverageWordLengthFeatureGenerator(BaseFeatureGenerator):
    """Generates average word length feature from email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}"
        
        # Split into words and calculate average length
        words = all_text.split()
        if not words:
            average_word_length = 0.0
        else:
            total_length = sum(len(word) for word in words)
            average_word_length = total_length / len(words)
        
        return {"average_word_length": average_word_length}
    
    @property
    def feature_names(self) -> list[str]:
        return ["average_word_length"]


class EmailEmbeddingsFeatureGenerator(BaseFeatureGenerator):
    """Generates embedding features using sentence transformers"""

    # Class-level model instance (loaded once and shared)
    _model = None

    @classmethod
    def _get_model(cls):
        """Lazy load the sentence transformer model"""
        if cls._model is None:
            # Use a small, efficient model (80MB, fast inference)
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body

        # Combine subject and body for embedding
        email_text = f"{subject} {body}"

        # Generate embedding using sentence transformer
        model = self._get_model()
        embedding = model.encode(email_text, convert_to_numpy=True)

        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()

        return {"average_embedding": embedding_list}

    @property
    def feature_names(self) -> list[str]:
        return ["average_embedding"]


class RawEmailFeatureGenerator(BaseFeatureGenerator):
    """Extracts raw email data as features"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        
        return {
            "email_subject": subject,
            "email_body": body
        }
    
    @property
    def feature_names(self) -> list[str]:
        return ["email_subject", "email_body"]


class NonTextCharacterFeatureGenerator(BaseFeatureGenerator):
    """Counts non-alphanumeric characters in email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}"
        
        # Count non-alphanumeric characters (excluding whitespace)
        non_text_char_count = sum(1 for char in all_text if not char.isalnum() and not char.isspace())
        
        return {"non_text_char_count": non_text_char_count}
    
    @property
    def feature_names(self) -> list[str]:
        return ["non_text_char_count"]