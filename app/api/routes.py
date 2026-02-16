from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.services.email_storage import EmailStorageService
from app.dataclasses import Email

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]
    classification_method: Optional[str] = None
    matched_email_id: Optional[int] = None
    similarity_score: Optional[float] = None

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

class TopicRequest(BaseModel):
    topic: str
    description: str

class TopicResponse(BaseModel):
    message: str
    topic: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None

class EmailStoreResponse(BaseModel):
    message: str
    email_id: int

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email)

        # Get classification metadata
        metadata = inference_service.model.get_classification_metadata(result["features"])

        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"],
            classification_method=metadata.get("method"),
            matched_email_id=metadata.get("matched_email_id"),
            similarity_score=metadata.get("similarity_score")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.post("/topics", response_model=TopicResponse)
async def add_topic(request: TopicRequest):
    """Add a new topic with description"""
    try:
        inference_service = EmailTopicInferenceService()

        # Add the topic using the model
        inference_service.model.add_topic(request.topic, request.description)

        return TopicResponse(
            message=f"Topic '{request.topic}' added successfully",
            topic=request.topic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/info")
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

@router.post("/emails/store", response_model=EmailStoreResponse)
async def store_email(request: EmailStoreRequest):
    """Store an email with optional ground truth label"""
    try:
        # Generate features for the email
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        features = inference_service.feature_factory.generate_all_features(email)

        # Extract the embedding from features
        embedding = features.get("email_embeddings_average_embedding")

        if embedding is None:
            raise HTTPException(status_code=400, detail="Failed to generate email embedding")

        # Store the email with its embedding
        storage_service = EmailStorageService()
        email_id = storage_service.save_email(
            subject=request.subject,
            body=request.body,
            embedding=embedding,
            ground_truth=request.ground_truth
        )

        return EmailStoreResponse(
            message="Email stored successfully",
            email_id=email_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO: LAB ASSIGNMENT - Part 2 of 2
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

