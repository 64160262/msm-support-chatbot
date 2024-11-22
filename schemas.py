from pydantic import BaseModel

class MsmRuleBase(BaseModel):
    keywords: str
    response: str

class MsmRuleCreate(MsmRuleBase):
    pass

class MsmRule(MsmRuleBase):
    id: int

    class Config:
        from_attributes = True 