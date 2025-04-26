import os, json, shutil

# 1) Create a new folder to hold per-business subfolders
os.makedirs("Dataset/photos_by_business", exist_ok=True)

# 2) Read your photos.json to know which business each photo belongs to
with open("Dataset/photos.json", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        bid = rec["business_id"]
        pid = rec["photo_id"]
        # assume your local image is at Dataset/photos/<photo_id>.jpg
        src = os.path.join("Dataset", "photos", f"{pid}.jpg")
        dst_dir = os.path.join("Dataset", "photos_by_business", bid)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, f"{pid}.jpg"))
