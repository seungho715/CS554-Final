import json
import torch
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from etl_business import Business, DATABASE_URL

class RecommendationEngine:
    def __init__(self, top_k=5):
        #Connect to PostgreSQL Database
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        self.session=Session()
        
        self.businesses = self.session.query(Business).all()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device = device)
        self.index = None
        self.business_embeddings = None
        self.build_index()
        self.top_k = top_k
    
    def build_corpus(self):
        corpus = []
        for business in self.businesses:
            categories = business.categories or ""
            if isinstance(categories, list):
                categories = ", ".join(categories)
            corpus.append(
                f"{business.name}. Categories: {categories}. "
                f"Located at: {business.address}, {business.city}, {business.state}."
            )
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
        print(f"Successfully built with {len(corpus)} items")
    
    def search(self, query, top_k=None):
        top_k = top_k or self.top_k
        query_embedding = np.array(self.model.encode([query]), dtype='float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.businesses[idx] for idx in indices[0]]
    
# if __name__=="__main__":
#     business_file_path = "../Dataset/yelp_academic_dataset_business.json"
#     engine = RecommendationEngine(business_file_path)
    
#     query = "I'm looking for a Mexican restaurant with great parking options in San Francisco."
#     results = engine.search(query)
    
#     print("Top recommendations:")
#     for business in results:
#         print(f"{business.get('name', 'Unknown')} - {business.get('address', '')}, {business.get('city', '')}")