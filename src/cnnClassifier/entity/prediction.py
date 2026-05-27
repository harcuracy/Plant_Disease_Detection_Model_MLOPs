from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    predicted_class: str = Field(..., alias="class")
    confidence: float
