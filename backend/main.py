import uuid
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from ultralytics import YOLO
from fastapi import HTTPException


yolo_model = YOLO("yolov8m-seg.pt")
YOLO_CONF_THRESHOLD = 0.5

inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
inpaint_pipeline.enable_model_cpu_offload()

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
    prompt: str


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/getImageSegments")
def get_image_segments(body: ImageSegmentRequestBody):
    if not body.url.startswith("https://"):
        raise HTTPException(400, 'URL should begin with "https://"')

    try:
        results = yolo_model.predict(body.url, conf=YOLO_CONF_THRESHOLD)
    except Exception as e:
        raise HTTPException(400, "Unable to download image from provided URL")

    objects = []
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            object_name = yolo_model.names[int(box.cls)]
            min_x = int(np.min(mask[:, 0]))
            max_x = int(np.max(mask[:, 0]))
            min_y = int(np.min(mask[:, 1]))
            max_y = int(np.min(mask[:, 1]))

            objects.append(
                {
                    "uuid": str(uuid.uuid4()),
                    "objectType": object_name,
                    "topLeft": {
                        "x": min_x,
                        "y": min_y,
                    },
                    "topRight": {
                        "x": max_x,
                        "y": min_y,
                    },
                    "bottomRight": {
                        "x": max_x,
                        "y": max_y,
                    },
                    "bottomLeft": {
                        "x": min_x,
                        "y": max_y,
                    },
                }
            )

    return {
        "objects": objects,
    }


@app.post("/inpaintImage")
async def inpaint(body: InpaintRequestBody):
    return {
        "url": "https://scholar.googleusercontent.com/citations?view_op=medium_photo&user=qIvZT74AAAAJ&citpid=7",
    }


@app.get("/test")
def test():
    init_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
    )
    mask_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/road-mask.png"
    )

    image = inpaint_pipeline(
        prompt="road", image=init_image, mask_image=mask_image
    ).images[0]
    image.save("/home/abhyudaya/MagicEraser/outputs/output1.png")
