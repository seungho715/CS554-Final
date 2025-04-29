from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dialogue.DialogueManager import DialogueManager
from recommendation.RecommendationEngine import RecommendationEngine
from recommendation.ReviewRetrieval import ReviewRetrieval
from ranker.Ranker import Ranker
from image.imagePipeline import ImagePipeline
import io
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from etl_business import Business, DATABASE_URL, Photo
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from uuid import uuid4


app = FastAPI()

conversations = {}

@app.post("/session")
def new_session():
    session_id = str(uuid4())
    conversations[session_id] = {
        "history": "",
        "slots": {},
        "awaiting_slot": None
    }
    return {"session_id": session_id}

# Initialize everything
dm = DialogueManager()
rec_engine = RecommendationEngine()
ranker = Ranker()
rr = ReviewRetrieval()

img_pipe = ImagePipeline(
    index_path="business_index.faiss",
    business_ids_path="business_ids.json"
)

db_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=db_engine)

# Request format
class RecommendIn(BaseModel):
    session_id: str
    user_input: str

# Endpoint
@app.post("/recommend")
def recommend(req: RecommendIn):
    state     = conversations.get(req.session_id)
    if not state:
        raise HTTPException(404, "Unknown session_id")
    history   = state["history"]
    slots     = state["slots"]
    awaiting  = state["awaiting_slot"]

    history += f"User: {req.user_input}\n"
    state["history"] = history
    if state["awaiting_slot"] == "cuisine":
        state["slots"]["cuisine"] = clean_cuisine(req.user_input)
        state["awaiting_slot"] = None
    elif state["awaiting_slot"] == "location":
        state["slots"]["location"] = req.user_input.strip().title()
        state["awaiting_slot"] = None
    else:
        new_slots = dm.process_conversation(state["history"])
        if "cuisine" in new_slots:
            new_slots["cuisine"] = clean_cuisine(new_slots["cuisine"])
            if "location" not in new_slots:
                new_slots["location"] = state["slots"].get("location")
        state["slots"].update({k:v for k,v in new_slots.items() if k in ("cuisine","location","recent_review_requested")})

    if "cuisine" not in state["slots"]:
        state["awaiting_slot"] = "cuisine"
        return {"ask": "Which cuisine are you interested in?"}
    if "location" not in state["slots"]:
        state["awaiting_slot"] = "location"
        return {"ask": "Which location are you looking in?"}

    query = f"{state['slots']['cuisine']} restaurants in {state['slots']['location']}"
    raw   = rec_engine.search(query)
    cands = [{
        "business_id": b.business_id,
        "name":        b.name,
        "address":     b.address,
        "city":        b.city,
        "state":       b.state,
        "categories":  b.categories or "",
        "stars":       b.stars
    } for b in raw]
    ranked = ranker.rerank(query, cands)

    if state["slots"].get("recent_review_requested"):
        revs = rr.get_recent_reviews(ranked[0]["business_id"])
        if revs:
            ranked[0]["review"] = revs[0].text

    return {"results": ranked}

def clean_cuisine(text: str) -> str:
    t = text.strip().lower()
    return t[:-1] if t.endswith("s") and len(t) > 1 else t

@app.post("/recommend/image")
async def image_recs(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    out = img_pipe.full_pipeline_image(img, top_k=5)
    business_ids_scores = out["top_businesses"]
    
    ids = [bid for bid, _ in business_ids_scores]
    with SessionLocal() as ses:
        rows = ses.execute(
            text(
                "SELECT business_id, name, address, city, state, stars "
                "FROM businesses WHERE business_id = ANY(:ids)"
            ),
            {"ids": ids},
        ).all()
        photos = {
            p.business_id: p
            for p in ses.query(Photo).filter(Photo.business_id.in_(ids)).all()
        }
    
    meta = {r.business_id: r for r in rows}
    
    results= []
    for bid, score in business_ids_scores:
        b = meta.get(bid)
        if not b:
            continue
        
        photo = photos.get(bid)
        url = None
        if photo:
            url = (
                f"http://{img_pipe.S3_BUCKET}.s3.amazonaws.com/"
                f"{bid}/{photo.photo_id}.jpg"
            )
        results.append({
            "business_id": bid,
            "name":        b.name,
            "address":     f"{b.address}, {b.city}, {b.state}",
            "stars":       b.stars,
            "score":       score,
            "image_url":   url,
        })
    
    return {
        "caption": out["caption"],
        "results": results
    }
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="path/to/frontend", html=True), name="frontend")