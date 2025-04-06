import json
import torch
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RecommendationEngine:
    def __init__(self, business_file_path):
        self.business_file_path = business_file_path
        self.businesses = self.load_business_data()
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(device = "cuda" if torch.cuda.is_available() else "cpu")
        self.index = None
        self.business_embeddings = None
        self.build_index()
    
    def load_business_data(self):
        #Loades the business data from the JSON file
        businesses = []
        with open(self.business_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    business = json.loads(line)
                    businesses.append(business)
                except json.JSONDecodeError:
                    continue
            return businesses
    
    def build_corpus(self):
        corpus = []
        for business in self.businesses:
            name = business.get("name", "")
            categories = business.get("categories", [])
            if isinstance(categories, list):
                categories = ", ".join(categories)
            else:
                categories = categories or ""
            address = business.get("address", "")
            city = business.get("city", "")
            state = business.get("state", "")
            doc = f"{name}. Categories: {categories}. Located at: {address}, {city}, {state}."
            corpus.append(doc)
        return corpus
    
    def build_index(self):
        #Use SentenceTransformer and builds FAISS index
        corpus = self.build_corpus()
        print("Embedding business data...")
        embeddings  =self.model.encode(corpus, show_progress_bar=True)
        self.business_embeddings = np.array(embeddings).astype('float32')
        dimension = self.business_embeddings.shape[1]
        print(f"Building FAISS Index with dimension {dimension}")
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.business_embeddings)
        print("Successfully built bitch")
    
    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            business = self.businesses[idx]
            results.append(business)
        return business
    
# if __name__=="__main__":
#     business_file_path = "../Dataset/yelp_academic_dataset_business.json"
#     engine = RecommendationEngine(business_file_path)
    
#     query = "I'm looking for a Mexican restaurant with great parking options in San Francisco."
#     results = engine.search(query)
    
#     print("Top recommendations:")
#     for business in results:
#         print(f"{business.get('name', 'Unknown')} - {business.get('address', '')}, {business.get('city', '')}")