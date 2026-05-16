from pydantic import BaseModel

class ProfileClient(BaseModel):
    level: str