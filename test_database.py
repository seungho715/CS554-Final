from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

load_dotenv()  # reads your .env

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # 1) count businesses
    cnt = conn.execute(text("SELECT COUNT(*) FROM businesses")).scalar()
    print("Businesses in DB:", cnt)

    # 2) sample a row
    row = conn.execute(text(
        "SELECT business_id, name, city, stars FROM businesses LIMIT 1"
    )).first()
    print("Sample row:", row)
