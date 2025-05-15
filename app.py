from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
ocr = PaddleOCR(use_angle_cls=True, lang="korean")

KST = timezone(timedelta(hours=9))


def response(status_code: int, *, error: str = None, data=None):
    return JSONResponse(
        status_code=200,
        content={
            "timestamp": datetime.now(KST).strftime('%Y-%m-%d %H:%M'),
            "status": status_code,
            "message": error,
            "data": data,
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, e: HTTPException):
    return response(e.status_code, error=e.detail)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, e: RequestValidationError):
    error_detail = e.errors()[0].get("msg", "Validation error")
    return response(400, error=error_detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, e: Exception):
    return response(500, error=str(e))


@app.post("/")
async def ocr_endpoint(request: Request):
    image_bytes = await request.body()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise Exception("이미지를 디코딩할 수 없습니다.")

    ocr_result = ocr.ocr(image, cls=True)[0]

    if ocr_result is None:
        return response(200, data=[])

    return response(200, data=list(map(lambda x: {
        'text': x[1][0],
        'score': x[1][1],
        'box': [{"x": int(point[0]), "y": int(point[1])} for point in x[0]]
    }, ocr_result)))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
