import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class imageCaptioner:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

    def caption(self, img_url):
        text = "a photo of"
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = self.processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)
        out = self.model.generate(**inputs)
        print(self.processor.decode(out[0], skip_special_tokens=True))
        return self.processor.decode(out[0], skip_special_tokens=True)

im = imageCaptioner()
im.caption('https://s3-media0.fl.yelpcdn.com/bphoto/PQirALJA6ugdTJT6KmmCuA/348s.jpg')
im.caption('https://s3-media0.fl.yelpcdn.com/bphoto/YG2rLCdfCYqQKlbfbtD3TA/o.jpg')

