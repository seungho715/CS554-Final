#Use Cross Encoder?from sentence_transformers import CrossEncoder
import torch

class Ranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def rerank(self, query, candidates, reviews=None):
        # reviews: a dict of business_id -> review text
        pairs = []
        for c in candidates:
            review_text = ""
            if reviews:
                review_text = reviews.get(c.get("business_id"), "")
            input_text = self._format_candidate(c, review_text)
            pairs.append((query, input_text))

        scores = self.model.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rank_score"] = float(scores[i])

        return sorted(candidates, key=lambda x: x["rank_score"], reverse=True)

    def _format_candidate(self, business, review_text=""):
        name = business.get("name", "")
        categories = business.get("categories", "")
        address = business.get("address", "")
        city = business.get("city", "")
        state = business.get("state", "")
        stars = business.get("stars", "")
        
        summary = f"{name}. Categories: {categories}. Located at: {address}, {city}, {state}. Rating: {stars} stars."
        if review_text:
            summary += f" Recent review: {review_text}"
        return summary


# ranker = Ranker()

# # Optional: gather reviews for top-k businesses
# rr = ReviewRetrieval("Dataset/yelp_review_mini.json")
# review_texts = {}

# for b in recommendations:
#     business_id = b.get("business_id")
#     reviews = rr.get_recent_reviews(business_id, top_n=1)
#     if reviews:
#         review_texts[business_id] = reviews[0].get("text", "")

# reranked = ranker.rerank(search_query, recommendations, reviews=review_texts)
