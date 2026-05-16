from pydantic import BaseModel

class Recommendation(BaseModel):
    action: str
    status: str