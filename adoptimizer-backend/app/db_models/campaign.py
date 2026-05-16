from pydantic import BaseModel

class Campaign(BaseModel):
    campaign_id: str
    platform: str