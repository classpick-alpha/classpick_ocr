from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang="korean")

KST = timezone(timedelta(hours=9))


@app.post('/')
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        response = []
        for item in ocr.ocr(image, cls=True)[0]:
            box_points, (text, score) = item
            response.append({
                "text": text,
                "score": score,
                "box": [{"x": int(point[0]), "y": int(point[1])} for point in box_points],
            })

        return JSONResponse(status_code=200, content={
            "timestamp": datetime.now(KST).strftime('%Y-%m-%d %H:%M'),
            "status": 200,
            "message": None,
            "data": response,
        })
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "timestamp": datetime.now(KST).strftime('%Y-%m-%d %H:%M'),
            "status": 500,
            "message": str(e),
            "data": None,
        })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
