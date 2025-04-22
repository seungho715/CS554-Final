from sqlalchemy import create_engine, Column, String, Float, Integer, JSON, Text, ForeignKey, PrimaryKeyConstraint, text, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
import json

# Define the ORM base class
Base = declarative_base()

#############################
# Define the Business Model #
#############################
class Business(Base):
    __tablename__ = 'businesses'
    
    business_id = Column(String, primary_key=True)  # Unique identifier
    name = Column(String)
    address = Column(String)
    city = Column(String)
    state = Column(String)
    postal_code = Column(String)  # Ensure this key matches your JSON (underscores, not space)
    latitude = Column(Float)
    longitude = Column(Float)
    stars = Column(Float)
    review_count = Column(Integer)
    is_open = Column(Integer)
    attributes = Column(JSON)  # Stores nested object data
    categories = Column(Text)  # Stored as a comma-separated string
    hours = Column(JSON)  # Hours stored as JSON

############################
# Define the Review Model  #
############################
class Review(Base):
    __tablename__ = 'reviews'
    
    review_id = Column(String, primary_key=True)
    user_id = Column(String)
    business_id = Column(String, ForeignKey('businesses.business_id'))  # Could add a ForeignKey constraint referencing businesses.business_id
    stars = Column(Float)
    date = Column(String)        # Optionally use Date type if desired
    text = Column(Text)
    useful = Column(Integer)
    funny = Column(Integer)
    cool = Column(Integer)
    
############################
# Define the Photo Model  #
############################   
class Photo(Base):
    __tablename__ = 'photos'
    
    photo_id = Column(String)
    business_id = Column(String, ForeignKey('businesses.business_id'))
    caption = Column(Text)
    label = Column(String)
    __table_args__ = (
        PrimaryKeyConstraint('photo_id', 'label'),
    )

#############################
# Database Setup and ETL    #
#############################
DATABASE_URL = "postgresql://postgres:qaz123@localhost:5433/yelp_db"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def create_tables():
    Base.metadata.create_all(engine)
    print("Tables created successfully.")
    
def truncate_table(table):
    session = Session()
    session.execute(text(f'TRUNCATE TABLE {table.__tablename__} RESTART IDENTITY;'))
    session.commit()
    session.close()
    print(f"Table {table.__tablename__} truncated.")

def load_business_data(business_file_path):
    session = Session()
    count = 0
    with open(business_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                business = Business(
                    business_id=data.get("business_id"),
                    name=data.get("name"),
                    address=data.get("address"),
                    city=data.get("city"),
                    state=data.get("state"),
                    postal_code=data.get("postal_code"),
                    latitude=data.get("latitude"),
                    longitude=data.get("longitude"),
                    stars=data.get("stars"),
                    review_count=data.get("review_count"),
                    is_open=data.get("is_open"),
                    attributes=data.get("attributes"),
                    categories=", ".join(data.get("categories", [])) if isinstance(data.get("categories"), list) else data.get("categories"),
                    hours=data.get("hours")
                )
                session.add(business)
                count += 1
                if count % 1000 == 0:
                    session.commit()
                    print(f"Committed {count} business records...")
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
                continue
    session.commit()
    session.close()
    print(f"Loaded {count} businesses into the database.")

def load_review_data(review_file_path):
    session = Session()
    count = 0
    with open(review_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                review = Review(
                    review_id=data.get("review_id"),
                    user_id=data.get("user_id"),
                    business_id=data.get("business_id"),
                    stars=data.get("stars"),
                    date=data.get("date"), 
                    text=data.get("text"),
                    useful=data.get("useful"),
                    funny=data.get("funny"),
                    cool=data.get("cool")
                )
                session.add(review)
                count += 1
                if count % 1000 == 0:
                    session.commit()
                    print(f"Committed {count} review records...")
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
                continue
    session.commit()
    session.close()
    print(f"Loaded {count} reviews into the database.")
    
def load_photos_data(photo_file_path):
    session = Session()
    count = 0
    with open(photo_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                photos = Photo(
                    photo_id=data.get("photo_id"),
                    business_id=data.get("business_id"),
                    caption=data.get("caption"),
                    label=data.get("label")
                )
                session.add(photos)
                count+=1
                if count % 1000==0:
                    session.commit()
                    print(f"Committed {count} photos records...")
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
                continue
    session.commit()
    session.close()
    print(f"Loaded {count} photos into the database.")


if __name__ == "__main__":
    create_tables()
    #truncate_table(Photo)
    business_file_path = "Dataset/yelp_academic_dataset_business.json"
    load_business_data(business_file_path)

    review_file_path = "Dataset/yelp_academic_dataset_review.json"
    load_review_data(review_file_path)
    
    photo_file_path = "Dataset/photos.json"
    
    load_photos_data(photo_file_path)
