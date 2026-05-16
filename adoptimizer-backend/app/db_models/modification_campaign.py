from pydantic import BaseModel

class ModificationCampaign(BaseModel):
    campaign_id: str
    modification_type: str