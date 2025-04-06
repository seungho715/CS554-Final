import json
from datetime import datetime

class ReviewRetrieval:
    def __int__(self, review_file_path):
        self.review_file_path = review_file_path
        self.review_by_business = self.load_reviews()
        
    def load_reviews(self):
        reviews_by_business = {}
        with open(self.review_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    business_id = review.get("business_id")
                    if business_id:
                        reviews_by_business.setdefault(business_id, []).append(review)
                except json.JSONDecodeError:
                    continue
            
        for business_id, reviews in reviews_by_business.items():
            reviews_by_business[business_id] = sorted(
                reviews,
                key=lambda r:datetime.strptime(r.get("date", "1900-01-01"), "%Y-%m-%d"),
                reverse=True
            )
        return reviews_by_business
    
    def get_recent_reviews(self, business_id, top_n=1):
        reviews = self.reviews_by_business.get(business_id, [])
        return reviews[:top_n]