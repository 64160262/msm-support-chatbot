from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL connection configuration
# DATABASE_URL = "mysql+pymysql://username:password@localhost:3306/msm_chatbot"
DATABASE_URL = "mysql+pymysql://root@localhost:3306/msm_support"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 