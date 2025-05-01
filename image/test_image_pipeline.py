# test_image_pipeline.py
from dotenv import load_dotenv
from image.imagePipeline import ImagePipeline

load_dotenv()   # so DATABASE_URL, AWS_*, etc. are loaded

# 1) Instantiate with your freshly-built files
pipe = ImagePipeline(
    index_path="business_index.faiss",
    business_ids_path="business_ids.json"
)

# 2) Pick a real (business_id, photo_id) from your database
#    (for example, query your `photos` table for one sample row)
sample_business_id = "Nk-SJhPlDBkAZvfsADtccA"
sample_photo_id    = "zsvj7vloL4L5jhYyPIuVwg"

# 3) Test captioning
caption = pipe.caption(sample_business_id, sample_photo_id)
print("Caption:", caption)

# 4) Test search
results = pipe.search(sample_business_id, sample_photo_id, top_k=5)
print("Top 5 similar businesses:", results)
