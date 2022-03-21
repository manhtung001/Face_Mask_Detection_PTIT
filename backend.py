import io
import uvicorn
import numpy as np
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
import os
import shutil
from PIL import Image

from utils import *

app = FastAPI(title='Face Mask Detection API')

# By using @app.get("/") you are allowing the GET method to work for the / endpoint.


dir_path = os.path.dirname(os.path.realpath(__file__))
tmpPath = os.path.join(dir_path, 'tmp')
if os.path.exists(tmpPath):
    shutil.rmtree(tmpPath)
if not os.path.exists(tmpPath):
    os.mkdir(tmpPath)


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Author: Tung Khong Manh. Now head over to " \
           "/docs. "


@app.post("/uploadImg")
async def uploadImg(fileUpload: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    filename = fileUpload.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    file_location = f"tmp/{fileUpload.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(fileUpload.file.read())
    print(f"info: file {fileUpload.filename} saved at {file_location}")

    labelList, imgResPath = predictImg(file_location)

    imgRes = Image.open(imgResPath, mode='r')

    bytes_image = io.BytesIO()
    imgRes.save(bytes_image, format='PNG')

    return Response(content=bytes_image.getvalue(), headers={"resLabel": str(labelList)}, media_type="image/png")

    # if os.path.exists(imgResPath):
    #     return FileResponse(imgResPath)

@app.post("/uploadVideo")
async def uploadVideo(fileUpload: UploadFile = File(...)):
    file_location = f"tmp/{fileUpload.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(fileUpload.file.read())
    print(f"info: file {fileUpload.filename} saved at {file_location}")

    # check video that hay o
    # checkIdentify = checkIdeViaVideo(file_location, uuid)

    return {
        "result": fileUpload.filename,
    }



# Allows the server to be run in this interactive environment
nest_asyncio.apply()

# Host depends on the setup you selected (docker or virtual env)
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

# Spin up the server!
uvicorn.run(app, host=host, port=5000)

