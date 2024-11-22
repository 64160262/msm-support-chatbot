from sqlalchemy import Column, Integer, String, Text
from database import Base

class MsmRule(Base):
    __tablename__ = "msm_rules"

    id = Column(Integer, primary_key=True, index=True)
    keywords = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
