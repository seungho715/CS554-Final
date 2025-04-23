from dialogue.DialogueManager import DialogueManager
from recommendation.RecommendationEngine import RecommendationEngine
from recommendation.ReviewRetrieval import ReviewRetrieval
from ranker.Ranker import Ranker

from sqlalchemy.orm import sessionmaker
from sqlalchemy     import create_engine
from etl_business   import Business, DATABASE_URL

def clean_cuisine(text: str) -> str:
    t = text.strip().lower()
    return t[:-1] if t.endswith("s") and len(t) > 1 else t

def main():
    dm = DialogueManager()
    rec_engine = RecommendationEngine()
    ranker = Ranker()
    rr = ReviewRetrieval()

    db_engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=db_engine)
    session = Session()

    rows = session.query(Business.categories).distinct().all()
    ALL_CUISINES = set()
    for (cats,) in rows:
        if not cats:
            continue
        for c in cats.split(","):
            c = c.strip().lower()
            if c.endswith("s"):
                c = c[:-1]
            ALL_CUISINES.add(c)

    history = ""
    slots = {}
    awaiting_slot = None

    while True:
        user = input("User: ")
        if user.lower().startswith(("quit", "exit", "thanks")):
            print("Assistant: Happy dining!")
            break

        # always append to history
        history += f"User: {user}\n"

        # 2) If we're waiting for a follow-up, consume that slot and go next turn
        if awaiting_slot == "cuisine":
            slots["cuisine"] = clean_cuisine(user)
            awaiting_slot = None
            continue

        if awaiting_slot == "location":
            slots["location"] = user.strip().title()
            awaiting_slot = None
            continue

        # 3) Quick keyword fallback for cuisine before LLM
        if not slots.get("cuisine"):
            for c in ALL_CUISINES:
                if c in user.lower():
                    slots["cuisine"] = c
                    break

        # 4) LLM extraction for any slots
        new_slots = dm.process_conversation(history)
        if new_slots.get("cuisine") is not None:
            new_slots["cuisine"] = clean_cuisine(new_slots["cuisine"])
        for k in ("cuisine", "location", "recent_review_requested", "other_info"):
            if new_slots.get(k) is not None:
                slots[k] = new_slots[k]

        # 5) If user said "review" and we have both cuisine+location, set flag
        if slots.get("cuisine") and slots.get("location"):
            if "review" in user.lower():
                slots["recent_review_requested"] = True

        # 6) Clarify missing cuisine
        if not slots.get("cuisine"):
            prompt = "Which cuisine are you interested in?"
            print("Assistant:", prompt)
            history += f"Assistant: {prompt}\n"
            awaiting_slot = "cuisine"
            continue

        # 7) Clarify missing location
        if not slots.get("location"):
            prompt = "Which location are you looking in?"
            print("Assistant:", prompt)
            history += f"Assistant: {prompt}\n"
            awaiting_slot = "location"
            continue

        # 8) Both slots filled → run recommendation + ranking
        query = f"{slots['cuisine']} restaurants in {slots['location']}"
        raw   = rec_engine.search(query)
        cands = [
            {
                "business_id": b.business_id,
                "name":        b.name,
                "address":     b.address,
                "city":        b.city,
                "state":       b.state,
                "categories":  b.categories,
                "stars":       b.stars
            }
            for b in raw
        ]
        ranked = ranker.rerank(query, candidates=cands)

        print("\nAssistant: Here are some options:")
        for r in ranked:
            print(f" • {r['name']} — {r['address']}, {r['city']} (score {r['rank_score']:.2f})")

        # 9) If a review was requested, fetch & show it
        if slots.get("recent_review_requested") and ranked:
            top = ranked[0]
            print(f"\nAssistant: Fetching the most recent review for {top['name']}…")
            reviews = rr.get_recent_reviews(top["business_id"], top_n=1)
            if reviews:
                print(f"[{reviews[0].date}] {reviews[0].text}")
            else:
                print("Assistant: Sorry, no reviews found.")

        # (Loop continues, allowing further refinements)

    session.close()

if __name__ == "__main__":
    main()
