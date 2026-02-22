from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Email:
    """Dataclass representing an email with subject and body"""
    subject: str
    body: str

@dataclass
class StoredEmail:
    """Email with ground truth label and pre-computed embedding"""
    id: int
    subject: str
    body: str
    ground_truth: Optional[str]
    embedding: List[float]
    timestamp: str