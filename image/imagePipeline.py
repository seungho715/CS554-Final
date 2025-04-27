import os
import json
import requests
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import faiss
from PIL import Image
from transformers import (CLIPProcessor, CLIPModel)

from imageCaptioner import *

from businessSearch import *

def img_from_disk(path):
    return Image.open(path).convert("RGB")

def img_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

# Pipeline combining FAISS business search and image captioner+classifier together
class imagePipeline:
    def __init__(self, index_path=None, business_ids_path=None, model_name="openai/clip-vit-base-patch32"):
        self.captioner = imageCaptioner()
        self.searcher = None

        if index_path and business_ids_path:
            self.load_index(index_path, business_ids_path, model_name)

    def _load(self, src):
        if isinstance(src, Image.Image):
            return src
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            return img_from_url(src)
        if isinstance(src, str) and os.path.exists(src):
            return img_from_disk(src)
        raise ValueError(f"Cannot load image from {src}")

    @staticmethod
    def make_faiss_index(json_path, image_dir, index_path, business_ids_path,
                         batch_size=64, model_name="openai/clip-vit-base-patch32"):
        idxr = businessIndexer(model_name=model_name)
        idxr.build(json_path, image_dir, batch_size=batch_size)
        idx, bids, *_ = idxr.get_index()

        faiss.write_index(idx, index_path)
        with open(business_ids_path, 'w') as f:
            json.dump(bids, f)
        print(f"FAISS index → {index_path}")
        print(f"Business IDs → {business_ids_path}")

    def load_index(self, index_path, business_ids_path, model_name):
        self.index = faiss.read_index(index_path)
        with open(business_ids_path, 'r') as f:
            bids = json.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        self.searcher = businessSearch(self.index, bids, model, processor, device)
        print(f"Loaded FAISS index (n_businesses={len(bids)})")

    def caption(self, src) -> str:
        img = self._load(src)
        return self.captioner.caption(img)

    def search(self, src, top_k=5):
        if not self.searcher:
            raise ValueError("Index not loaded.")
        img = self._load(src)
        return self.searcher.query(img, top_k)

    def full_pipeline(self, src, top_k=5):
        img = self._load(src)
        return {
            "caption": self.captioner.caption(img),
            "top_businesses": self.searcher.query(img, top_k)
        }

# For making the initial FAISS index
if __name__ == "__main__":

    photo_json = "../Dataset/Yelp_Photos/photos.json"
    photo_folder = "../Dataset/Yelp_Photos/photos"

    imagePipeline.make_faiss_index(photo_json, photo_folder, "business_index.faiss", "business_ids.json")

    '''
    pipeline = imagePipeline(
        index_path="business_index.faiss",
        business_ids_path="business_ids.json"
    )

    while True:
        inp = input("Enter URL or file path: ")
        out = pipeline.full_pipeline(inp)
        print("Caption:       ", out["caption"])
        print("Top Businesses:", out["top_businesses"])
        print()
    '''