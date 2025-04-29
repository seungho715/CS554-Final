# image/businessSearch.py
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

def compute_embeddings_batch(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    return embs.cpu().numpy()

class businessIndexer:
    """
    Incrementally build a FAISS index of per-business averaged CLIP embeddings
    via repeated add_batch() calls, then finalize with get_index().
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        D = self.model.config.projection_dim
        self._sums   = defaultdict(lambda: np.zeros(D, dtype=np.float32))
        self._counts = defaultdict(int)

    def add_batch(self, images, business_ids):
        """
        Embed a batch of PIL images & accumulate sums/counts per business.
        images: List[PIL.Image], business_ids: List[str]
        """
        embs = compute_embeddings_batch(images, self.model, self.processor, self.device)
        for emb, bid in zip(embs, business_ids):
            self._sums[bid]   += emb
            self._counts[bid] += 1

    def get_index(self):
        """
        After all add_batch() calls, returns:
          - faiss.IndexFlatIP of per-business mean embeddings
          - list of business_ids in the same order as the faiss rows
          - (optionally) model, processor, device for downstream search
        """
        bids = list(self._sums.keys())
        mat  = np.vstack([ self._sums[bid] / self._counts[bid] for bid in bids ])
        faiss.normalize_L2(mat)

        dim   = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        return index, bids, self.model, self.processor, self.device


class businessSearch:
    """
    Given a FAISS index + ordered business_ids + a CLIP model, query nearest neighbors of an image.
    """
    def __init__(self, index, business_ids, model, processor, device):
        self.index       = index
        self.business_ids = business_ids
        self.model       = model
        self.processor   = processor
        self.device      = device

    def query(self, img, top_k=5):
        embs = compute_embeddings_batch([img], self.model, self.processor, self.device)
        faiss.normalize_L2(embs)
        D, I = self.index.search(embs.astype('float32'), top_k)
        return [ self.business_ids[i] for i in I[0] ]
