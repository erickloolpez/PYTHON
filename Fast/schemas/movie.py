
from pydantic import BaseModel, Field
from typing import Optional 

class Movie(BaseModel):
    id: Optional[int] = None
    title: str = Field(default="My pelicula", min_length=5, max_length=15)
    overview: str
    year: int
    rating: float
    category: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "title": "Perro",
                "overview": "Es un lindo perro.",
                "year": 2022,
                "rating": 8,
                "category": "Love"
            }
        }
