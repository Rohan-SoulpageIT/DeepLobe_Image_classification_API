from PIL import Image
import os
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
import uvicorn
import io
from starlette.config import Config
from starlette.templating import Jinja2Templates
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from classifier import *

# getting all the templets for the following dir.
templates = Jinja2Templates(directory="templates")


async def image_classification_upload_page(request):
    if request.method == "POST":
        form = await request.form()
        file = form["image"].file
        # getting image data.
        image = io.BytesIO(file.read())

        start = time.time()
        # preprocessing image data.
        pre_processed_image = pre_processing(image)

        # getting class of the image.
        class_of_image = classifiy_image(pre_processed_image)
        end = time.time()
        if file:
            context = {
                "class_of_image": class_of_image,
                "Time_of_execution_in_seconds": str(round(end - start, 3)),
            }
            return JSONResponse(context)
    else:
        return templates.TemplateResponse(
            "image_classification.html", {"request": request, "data": ""}
        )


# All the routs of this website.
routes = [
    Route(
        "/Deeplobe.ai-Image-classification",
        image_classification_upload_page,
        methods=["GET", "POST"],
    ),
]
# App congiguration.
app = Starlette(
    debug=True,
    routes=routes,
)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
