# build_image_index.py
import os
from dotenv import load_dotenv

# this import path must point at wherever you saved the code above
from image.imagePipeline import ImagePipeline  

load_dotenv()   # pulls in DATABASE_URL, AWS_* from your .env

# one‐time: build and serialize your FAISS index
ImagePipeline.build_index_from_db(
    index_path="business_index.faiss",
    business_ids_path="business_ids.json",
    batch_size=64,
    clip_model_name="openai/clip-vit-base-patch32",
)
