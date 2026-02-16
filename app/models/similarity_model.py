import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

class EmailClassifierModel:
    """Email classifier model using embedding similarity"""

    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())

        # Load sentence transformer model (same model as feature generator)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-compute embeddings for all topic descriptions
        self.topic_embeddings = self._compute_topic_embeddings()
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)

    def _compute_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all topic descriptions"""
        topic_embeddings = {}
        for topic, data in self.topic_data.items():
            description = data['description']
            embedding = self.model.encode(description, convert_to_numpy=True)
            topic_embeddings[topic] = embedding
        return topic_embeddings
    
    def predict(self, features: Dict[str, Any]) -> str:
        """
        Classify email using similarity if emails exist, else use topics
        Returns the predicted topic/label
        """
        email_embedding = features.get("email_embeddings_average_embedding", None)
        
        if email_embedding is None:
            return "unknown"
        
        # Convert to numpy array if it's a list
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)
        
        # Try similarity-based classification first
        prediction, _ = self._classify_by_similarity(email_embedding)
        
        if prediction:
            return prediction
        
        # Fallback to topic-based classification
        return self._classify_by_topics(features)
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Get classification scores for all topics
        Note: This always returns topic-based scores, even if similarity classification is used
        """
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def get_classification_metadata(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about how the classification was performed
        Returns method used and relevant details
        """
        email_embedding = features.get("email_embeddings_average_embedding", None)
        
        if email_embedding is None:
            return {"method": "unknown"}
        
        # Convert to numpy array if it's a list
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)
        
        # Check if similarity-based classification would be used
        _, metadata = self._classify_by_similarity(email_embedding)
        
        if metadata:
            return metadata
        
        return {"method": "topic"}
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate cosine similarity between email and topic embeddings"""
        # Get email embedding from features (now a list/array)
        email_embedding = features.get("email_embeddings_average_embedding", None)

        if email_embedding is None:
            return 0.0

        # Convert to numpy array if it's a list
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        # Get pre-computed topic embedding
        topic_embedding = self.topic_embeddings[topic]

        # Calculate cosine similarity
        # cosine_similarity = dot(A, B) / (||A|| * ||B||)
        dot_product = np.dot(email_embedding, topic_embedding)
        email_norm = np.linalg.norm(email_embedding)
        topic_norm = np.linalg.norm(topic_embedding)

        if email_norm == 0 or topic_norm == 0:
            return 0.0

        cosine_similarity = dot_product / (email_norm * topic_norm)

        # Cosine similarity is between -1 and 1, but for text it's usually positive
        # Normalize to 0-1 range for better interpretability
        normalized_score = (cosine_similarity + 1) / 2

        return float(normalized_score)
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}
    
    def _load_stored_emails(self) -> List[Dict[str, Any]]:
        """Load stored emails with embeddings from data/emails.json"""
        emails_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
        
        if not os.path.exists(emails_file):
            return []
        
        try:
            with open(emails_file, 'r') as f:
                emails = json.load(f)
                return emails if isinstance(emails, list) else []
        except (json.JSONDecodeError, IOError):
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _classify_by_similarity(self, email_embedding: np.ndarray) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Find most similar stored email and return its ground truth
        
        Returns:
            Tuple of (predicted_label, metadata_dict)
            If no match found, returns (None, {})
        """
        stored_emails = self._load_stored_emails()
        
        if not stored_emails:
            return None, {}
        
        max_similarity = -1
        best_match = None
        
        for stored_email in stored_emails:
            # Skip emails without ground truth
            if not stored_email.get('ground_truth'):
                continue
            
            stored_embedding = np.array(stored_email['embedding'])
            similarity = self._cosine_similarity(email_embedding, stored_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = stored_email
        
        if best_match:
            return best_match['ground_truth'], {
                "method": "similarity",
                "matched_email_id": best_match['id'],
                "similarity_score": float(max_similarity)
            }
        
        return None, {}
    
    def _classify_by_topics(self, features: Dict[str, Any]) -> str:
        """Original topic-based classification logic"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        return max(scores, key=scores.get)
    
    def add_topic(self, topic_name: str, description: str):
        """
        Add a new topic and recompute embeddings
        
        Args:
            topic_name: Name of the new topic
            description: Description for the topic
        """
        # Load current topics from file
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        
        with open(data_file, 'r') as f:
            topic_data = json.load(f)
        
        # Add new topic
        topic_data[topic_name] = {"description": description}
        
        # Save to file
        with open(data_file, 'w') as f:
            json.dump(topic_data, f, indent=2)
        
        # Reload and recompute embeddings
        self.topic_data = topic_data
        self.topics = list(topic_data.keys())
        self.topic_embeddings = self._compute_topic_embeddings()