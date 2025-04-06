from dialogue.DialogueManager import DialogueManager
from recommendation.RecommendationEngine import RecommendationEngine
from recommendation.ReviewRetrieval import ReviewRetrieval

def main():
    # Initialize the dialogue manager
    dm = DialogueManager()
    
    conversation = (
        "User: I'm looking for a Chinese restaurant with great parking options in San Francisco, "
        "and can you show me a recent review of the place?"
    )
    
    query_data = dm.process_conversation(conversation)
    search_query = query_data.get("query", "Chinese restaurant in San Francisco")
    recent_review_flag = query_data.get("recent_review_requested", False)
    
    business_file_path = "../Dataset/yelp_academic_dataset_business.json"
    engine = RecommendationEngine(business_file_path)
    
    recommendations = engine.search(search_query)
    
    print("Top recommendations:")
    for business in recommendations:
        print(f"{business.get('name', 'Unknown')} - {business.get('address', '')}, {business.get('city', '')}")
    
    if query_data.get("recent_review_requested"):
        print("Retrieving a recent review for the selected business...")
    
    if recent_review_flag and recommendations:
        selected_business = recommendations[0]
        business_id = selected_business.get("business_id")
        review_file_path = "../Dataset/yelp_academic_dataset_review.json"
        rr = ReviewRetrieval(review_file_path)
        recent_reviews = rr.get_recent_reviews(business_id, top_n=1)
        
        print("\nRecent Review for", selected_business.get("name", "Unknown"), ":")
        for review in recent_reviews:
            print(review.get("text", "No review text available."))

if __name__ == "__main__":
    main()