from pydantic import BaseModel

class Metric(BaseModel):
    ctr: float
    cpc: float
    roas: float