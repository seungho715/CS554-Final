from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from etl_business import Business, DATABASE_URL

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

first_ten = session.query(Business).limit(10).all()

for biz in first_ten:
    print(f"{biz.business_id} | {biz.name} | {biz.city}, {biz.state} | {biz.stars}")

session.close()