# from flask import Flask, render_template, request
# import os

# app = Flask(__name__)

# # Commenting out RecommendationEngine and ReviewRetrieval for demo
# # class RecommendationEngine:
# #     def __init__(self, top_k=5):
# #         self.top_k = top_k
    
# #     def search(self, user_input):
# #         # Return mock recommendations for demo purposes
# #         return [
# #             {'id': 1, 'name': 'The Italian Bistro', 'address': '123 Pasta Rd', 'city': 'Rome', 'rating': 4.5},
# #             {'id': 2, 'name': 'Sushi Master', 'address': '456 Sushi St', 'city': 'Tokyo', 'rating': 4.7},
# #             {'id': 3, 'name': 'Taco Heaven', 'address': '789 Taco Blvd', 'city': 'Mexico City', 'rating': 4.3},
# #         ]

# # class ReviewRetrieval:
# #     def get_recent_reviews(self, restaurant_id):
# #         # Return mock reviews for demo purposes
# #         return [{'text': f"Great food at restaurant {restaurant_id}!"}]

# # Initialize Flask app
# # recommendation_engine = RecommendationEngine(top_k=5)
# # review_retrieval = ReviewRetrieval()

# # Upload folder setup for image (although we're not processing the image in this demo)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/get_recommendations', methods=['POST'])
# def get_recommendations():
#     # Since RecommendationEngine is commented out, we will not process the user input or image for now
#     # Just return a mock recommendation as part of the demo
    
#     # Placeholder recommendations for demo purposes
#     recommendations = [
#         {'name': 'The Italian Bistro', 'address': '123 Pasta Rd', 'city': 'Rome', 'score': 4.5, 'review': 'Great food!'},
#         {'name': 'Sushi Master', 'address': '456 Sushi St', 'city': 'Tokyo', 'score': 4.7, 'review': 'Amazing sushi!'},
#         {'name': 'Taco Heaven', 'address': '789 Taco Blvd', 'city': 'Mexico City', 'score': 4.3, 'review': 'Best tacos in town!'}
#     ]
    
#     return render_template('index.html', recommendations=recommendations, user_input="Italian Food")

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
