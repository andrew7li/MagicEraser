import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageSegmentRequestBody(BaseModel):
    url: str


class InpaintRequestBody(BaseModel):
    url: str


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/getImageSegments")
async def get_image_segments(body: ImageSegmentRequestBody):
    return {
        "objects": [
            {
                "uuid": str(uuid.uuid4()),
                "objectType": "bus",
                "topLeft": {
                    "x": 50,
                    "y": 100,
                },
                "topRight": {
                    "x": 100,
                    "y": 100,
                },
                "bottomRight": {
                    "x": 100,
                    "y": 200,
                },
                "bottomLeft": {
                    "x": 50,
                    "y": 200,
                },
                "confidence": 0.5,
            },
            {
                "uuid": str(uuid.uuid4()),
                "objectType": "car",
                "topLeft": {
                    "x": 50,
                    "y": 100,
                },
                "topRight": {
                    "x": 100,
                    "y": 100,
                },
                "bottomRight": {
                    "x": 100,
                    "y": 200,
                },
                "bottomLeft": {
                    "x": 50,
                    "y": 200,
                },
                "confidence": 0.9,
            }
        ]
    }


@app.post("/inpaintImage")
async def inpaint(body: InpaintRequestBody):
    return {
        "url": "https://scholar.googleusercontent.com/citations?view_op=medium_photo&user=qIvZT74AAAAJ&citpid=7",
    }
