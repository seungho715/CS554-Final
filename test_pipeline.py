from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
dm = DialogueManager()
engine = RecommendationEngine()
ranker = Ranker()
rr = ReviewRetrieval()

class Query(BaseModel):
    text: str

@app.post("/recommend")
def recommend(q: Query):
    qd = dm.process_conversation(q.text)
    candidates = engine.search(qd['query'])
    ranked = ranker.rerank(qd['query'], [c.__dict__ for c in candidates])
    result = [ { 'name':b['name'], 'address':b['address'], 'score':b['rank_score'] }
               for b in ranked ]
    if qd.get('recent_review_requested') and result:
        reviews = rr.get_recent_reviews(ranked[0]['business_id'])
        result[0]['review'] = reviews[0].text if reviews else None
    return { 'results': result }
