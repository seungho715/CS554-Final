from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from etl_business import Review, DATABASE_URL

class ReviewRetrieval:
    def __init__(self):
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        self.session = Session()
    
    def get_recent_reviews(self, business_id, top_n=1):
        reviews = self.session.query(Review).filter(Review.business_id == business_id).order_by(desc(Review.date)).limit(top_n).all()
        return reviews