from fastapi import FastAPI
from pydantic import BaseModel
from dialogue.DialogueManager import DialogueManager
from recommendation.RecommendationEngine import RecommendationEngine
from recommendation.ReviewRetrieval import ReviewRetrieval
from ranker.Ranker import Ranker

app = FastAPI()

# Initialize everything
dm = DialogueManager()
engine = RecommendationEngine(business_file_path="Dataset/yelp_business_mini.json")
ranker = Ranker()
rr = ReviewRetrieval(review_file_path="Dataset/yelp_review_mini.json")

# Request format
class Query(BaseModel):
    text: str

# Endpoint
@app.post("/recommend")
def recommend(q: Query):
    qd = dm.process_conversation(q.text)
    
    candidates = engine.search(qd['query'])

    ranked = ranker.rerank(qd['query'], candidates)

    result = [
        {
            'name': b['name'],
            'address': b['address'],
            'score': b['rank_score']
        } for b in ranked
    ]

    if qd.get('recent_review_requested') and result:
        first_business_reviews = rr.get_recent_reviews(ranked[0]['business_id'])
        if first_business_reviews:
            result[0]['review'] = first_business_reviews[0]['text']

    return { 'results': result }
