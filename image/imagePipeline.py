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
from image.businessSearch import businessIndexer
from image.imageCaptioner import imageCaptioner

# ─── S3 helper ─────────────────────────────────────────────────────────────

s3 = boto3.client("s3")
BUCKET = "cs554-yelp-photos"

def _load_from_s3(business_id: str, photo_id: str) -> Image.Image:
    key = f"{business_id}/{photo_id}.jpg"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return Image.open(obj["Body"]).convert("RGB")

# ─── main pipeline ─────────────────────────────────────────────────────────

class ImagePipeline:
    S3_BUCKET = BUCKET
    @staticmethod
    def build_index_from_db(
        index_path:        str,
        business_ids_path: str,
        batch_size:        int = 64,
        clip_model_name:   str = "openai/clip-vit-base-patch32"
    ):
        """
        One‐off: Pull photo metadata from Postgres, images from S3, embed via CLIP,
        build FAISS index (one vector per business), and write it to disk.
        """
        # DB session
        engine  = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        photos  = session.query(Photo).all()

        # CLIP embedding setup
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        clip      = CLIPModel.from_pretrained(clip_model_name).to(device)
        processor = CLIPProcessor.from_pretrained(clip_model_name)

        # businessIndexer will average per-business
        idxr      = businessIndexer(model_name=clip_model_name)

        # batch & accumulate
        for i in range(0, len(photos), batch_size):
            batch = photos[i : i + batch_size]
            imgs, bids = [], []
            for p in batch:
                try:
                    img = _load_from_s3(p.business_id, p.photo_id)
                except Exception as err:
                    print(f"skipping corrupt image {p.photo_id}: {err}")
                    continue
                imgs.append(img)
                bids.append(p.business_id)

            if imgs:  
                idxr.add_batch(imgs, bids)

        # finalize & dump
        index, business_list, *_ = idxr.get_index()
        faiss.write_index(index, index_path)
        with open(business_ids_path, "w") as f:
            json.dump(business_list, f)

        print(f"Built FAISS index → {index_path}")
        print(f"Saved business IDs → {business_ids_path}")

    def __init__(
        self,
        index_path:        str,
        business_ids_path: str,
        clip_model_name:   str = "openai/clip-vit-base-patch32"
    ):
        # 1) load vector index + business ordering
        self.index = faiss.read_index(index_path)
        with open(business_ids_path, "r") as f:
            self.bids = json.load(f)

        # 2) CLIP setup
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.proc = CLIPProcessor.from_pretrained(clip_model_name)

        # 3) BLIP captioner
        self.captioner = imageCaptioner()
        self.S3_BUCKET = BUCKET

    def caption(self, business_id: str, photo_id: str) -> str:
        img = _load_from_s3(business_id, photo_id)
        return self.captioner.caption(img)

    def search(self, business_id: str, photo_id: str, top_k: int = 5):
        """
        Given (business_id,photo_id) as S3 keys, return the top_k closest
        businesses by CLIP similarity.
        """
        img    = _load_from_s3(business_id, photo_id)
        inputs = self.proc(images=[img], return_tensors="pt").to(self.clip.device)
        feat_tensor = self.clip.get_image_features(**inputs).cpu()
        feat   = feat_tensor.detach().numpy()
        D, I   = self.index.search(feat.astype("float32"), top_k)

        return [(self.bids[i], float(D[0][j])) for j, i in enumerate(I[0])]

    def full_pipeline(self, business_id: str, photo_id: str, top_k: int = 5):
        return {
            "caption":       self.caption(business_id, photo_id),
            "top_businesses": self.search(business_id, photo_id, top_k)
        }
        
    def caption_image(self, img: Image.Image) -> str:
        """
        Take a PIL image and return a BLIP caption string.
        """
        return self.captioner.caption(img)

    def search_image(self, img: Image.Image, top_k: int = 5):
        """
        Take a PIL image and return top_k (business_id, score) from FAISS.
        """
        # 1) tokenize & embed
        inputs = self.proc(images=[img], return_tensors="pt", padding=True).to(self.clip.device)
        with torch.no_grad():
            feats = self.clip.get_image_features(**inputs)

        # 2) move off GPU / strip grad
        feats_np = feats.detach().cpu().numpy().astype("float32")

        # 3) query FAISS
        D, I = self.index.search(feats_np, top_k)

        # 4) map back to business IDs
        return [
            (self.bids[i], float(D[0][j]))
            for j, i in enumerate(I[0])
        ]

    def full_pipeline_image(self, img: Image.Image, top_k: int = 5):
        """
        Combined caption + visual search
        """
        return {
            "caption":       self.caption_image(img),
            "top_businesses": self.search_image(img, top_k)
        }