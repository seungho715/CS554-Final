from sentence_transformers import CrossEncoder
import torch

class Ranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Load cross-encoder model
        self.model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def rerank(self, query, candidates):
        # Prepare (query, business_description) pairs
        pairs = [(query, self._format_candidate(b)) for b in candidates]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for i, candidate in enumerate(candidates):
            candidate["rank_score"] = float(scores[i])

        return sorted(candidates, key=lambda x: x["rank_score"], reverse=True)

    def _format_candidate(self, business):
        # Build a string from the business info
        name = business.get("name", "")
        categories = business.get("categories", "")
        address = business.get("address", "")
        city = business.get("city", "")
        state = business.get("state", "")
        stars = business.get("stars", "")
        
        return f"{name}. Categories: {categories}. Located at: {address}, {city}, {state}. Rating: {stars} stars."
