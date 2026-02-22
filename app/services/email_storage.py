import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

class EmailStorageService:
    """Service for managing stored emails with embeddings"""
    
    def __init__(self, data_file: str = None):
        if data_file is None:
            # Default path relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.data_file = os.path.join(base_dir, 'data', 'emails.json')
        else:
            self.data_file = data_file
    
    def _load_emails(self) -> List[Dict[str, Any]]:
        """Load emails from JSON file"""
        if not os.path.exists(self.data_file):
            return []
        
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted or empty, return empty list
            return []
    
    def _save_to_file(self, emails: List[Dict[str, Any]]):
        """Save emails to JSON file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        with open(self.data_file, 'w') as f:
            json.dump(emails, f, indent=2)
    
    def save_email(self, subject: str, body: str, embedding: List[float], 
                   ground_truth: Optional[str] = None) -> int:
        """
        Save a new email with its embedding to storage
        
        Args:
            subject: Email subject
            body: Email body
            embedding: Pre-computed embedding vector
            ground_truth: Optional ground truth label for classification
            
        Returns:
            The ID assigned to the stored email
        """
        emails = self._load_emails()
        email_id = len(emails) + 1
        
        email_entry = {
            "id": email_id,
            "subject": subject,
            "body": body,
            "embedding": embedding,
            "ground_truth": ground_truth,
            "timestamp": datetime.now().isoformat()
        }
        
        emails.append(email_entry)
        self._save_to_file(emails)
        
        return email_id
    
    def get_all_emails(self) -> List[Dict[str, Any]]:
        """Get all stored emails with embeddings"""
        return self._load_emails()
    
    def get_email_by_id(self, email_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific email by ID"""
        emails = self._load_emails()
        for email in emails:
            if email.get('id') == email_id:
                return email
        return None
    
    def count_emails(self) -> int:
        """Count total stored emails"""
        return len(self._load_emails())
