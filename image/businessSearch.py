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

from imagePipeline import img_from_disk

def load_metadata(json_path):
    with open(json_path, 'r') as f:
        for line in f:
            yield json.loads(line)

def compute_embeddings_batch(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    return embs.cpu().numpy()

# Create FAISS index of photos and businesses using CLIP embedder
class businessIndexer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index = None
        self.business_ids = None

    def build(self, json_path, image_dir, batch_size=64):
        D = self.model.config.projection_dim
        sums, counts = defaultdict(lambda: np.zeros(D, dtype=np.float32)), defaultdict(int)
        buf_imgs, buf_ids = [], []

        for rec in load_metadata(json_path):
            img_path = os.path.join(image_dir, f"{rec['photo_id']}.jpg")
            try:
                img = img_from_disk(img_path)
            except Exception as e:
                print(f"Skipping {rec['photo_id']}: {e}")
                continue

            buf_imgs.append(img); buf_ids.append(rec["business_id"])
            if len(buf_imgs) >= batch_size:
                embs = compute_embeddings_batch(buf_imgs, self.model, self.processor, self.device)
                for emb, bid in zip(embs, buf_ids):
                    sums[bid]  += emb
                    counts[bid] += 1
                buf_imgs.clear(); buf_ids.clear()

        if buf_imgs:
            embs = compute_embeddings_batch(buf_imgs, self.model, self.processor, self.device)
            for emb, bid in zip(embs, buf_ids):
                sums[bid]  += emb
                counts[bid] += 1

        self.business_ids = list(sums.keys())
        mat = np.vstack([sums[bid] / counts[bid] for bid in self.business_ids])
        faiss.normalize_L2(mat)
        self.index = faiss.IndexFlatIP(D)
        self.index.add(mat)

    def get_index(self):
        return self.index, self.business_ids, self.model, self.processor, self.device

# Query FAISS for similar business IDs based on image given
class businessSearch:
    def __init__(self, index, business_ids, model, processor, device):
        self.index     = index
        self.business_ids   = business_ids
        self.model     = model
        self.processor = processor
        self.device    = device

    def query(self, img: Image.Image, top_k=5):
        embs = compute_embeddings_batch([img], self.model, self.processor, self.device)
        faiss.normalize_L2(embs)
        _, indices = self.index.search(embs.astype('float32'), top_k)
        return [self.business_ids[i] for i in indices[0]]
