from pprint import pprint
import pandas as pd

import torch
import torch
from PIL import Image
from transformers import (
    pipeline, BlipProcessor, BlipForConditionalGeneration,
)
# Create CLIP classifications of which kind of image it is (based on what yelp does)
class imageClassifier:
    def __init__(self, device=0):
        self.clip = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            torch_dtype=torch.bfloat16,
            device=device,
        )
        self.labels = ['inside', 'outside', 'drink', 'food', 'menu']

    def classify(self, img: Image.Image):
        return self.clip(img, candidate_labels=self.labels)

# Caption the image with BLIP based on what the image is classified as, giving it a little push
class imageCaptioner:
    def __init__(self, device="cuda"):
        self.classifier = imageClassifier()
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16
        ).to(device)

    def caption(self, img: Image.Image) -> str:
        result = self.classifier.classify(img)
        label = result[0]['label']
        prefix_map = {
            'inside': "An inside image of",
            'outside': "An outside image of",
            'drink': "A drink image of",
            'food': "A food image of",
            'menu': "A menu image of",
        }
        prefix = prefix_map.get(label, "")

        inputs = self.processor(img, prefix, return_tensors="pt").to(self.model.device, torch.float16)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
