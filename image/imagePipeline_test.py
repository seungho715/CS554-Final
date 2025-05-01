import os
import json
import faiss
import boto3
import torch

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from transformers import CLIPModel, CLIPProcessor


from etl_business import Photo, DATABASE_URL
from image.imageCaptioner import imageCaptioner
from image.businessSearch import businessIndexer

# S3 client + bucket name
s3 = boto3.client("s3")
BUCKET = "cs554-yelp-photos"

class ImagePipeline:
    def __init__(self,
                 index_path: str,
                 business_ids_path: str,
                 clip_model_name="openai/clip-vit-base-patch32"):
        # 1) load FAISS index + business id order
        self.index = faiss.read_index(index_path)
        with open(business_ids_path, "r") as f:
            self.bids = json.load(f)

        # 2) set up CLIP for runtime queries
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.proc = CLIPProcessor.from_pretrained(clip_model_name)

        # 3) captioner
        self.captioner = imageCaptioner()

    def _load_from_s3(self, business_id: str, photo_id: str) -> Image.Image:
        key = f"{business_id}/{photo_id}.jpg"
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return Image.open(obj["Body"]).convert("RGB")

    def caption(self, business_id: str, photo_id: str) -> str:
        img = self._load_from_s3(business_id, photo_id)
        return self.captioner.caption(img)

    def search(self, business_id: str, photo_id: str, top_k=5):
        """
        Given an input image (as s3 keys), returns top_k businesses ranked by CLIP.
        """
        img = self._load_from_s3(business_id, photo_id)
        inputs = self.proc(images=img, return_tensors="pt").to(self.clip.device)
        feat = self.clip.get_image_features(**inputs).cpu().numpy()
        D, I = self.index.search(feat, top_k)
        return [(self.bids[i], float(D[0][j])) for j, i in enumerate(I[0])]

    @staticmethod
    def make_faiss_index_from_db(index_path: str,
                                 business_ids_path: str,
                                 batch_size=64,
                                 clip_model_name="openai/clip-vit-base-patch32"):
        """
        One-time: read ALL Photo rows from Postgres, fetch each image
        from S3, embed with CLIP, build FAISS index and save it.
        """
        # set up DB session
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        photos = session.query(Photo).all()
        
        # CLIP embedder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        proc = CLIPProcessor.from_pretrained(clip_model_name)

        idxr = businessIndexer(model_name=clip_model_name)

        # batch through every photo
        for i in range(0, len(photos), batch_size):
            batch = photos[i:i+batch_size]
            imgs, bids = [], []
            for p in batch:
                img = Image.open(
                    s3.get_object(Bucket=BUCKET,
                                  Key=f"{p.business_id}/{p.photo_id}.jpg")["Body"]
                ).convert("RGB")
                imgs.append(img)
                bids.append(p.business_id)

            idxr.add_batch(imgs, bids)  # whatever your indexer API is

        # extract FAISS index + business order
        idx, business_list, *_ = idxr.get_index()
        faiss.write_index(idx, index_path)
        with open(business_ids_path, "w") as f:
            json.dump(business_list, f)

        print(f"Built FAISS index → {index_path}")
        print(f"Saved business IDs → {business_ids_path}")
