from typing import Dict, List

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import logging

from tagging_utils import predict_tags_dgb

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/tagging")
async def process_image(file: UploadFile) -> Dict[str, List[str]]:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    tags = predict_tags_dgb(image)

    return JSONResponse(
        status_code=200,
        content={
            "tags": tags,
            "space_line": " ".join(tags),
        }
    )

