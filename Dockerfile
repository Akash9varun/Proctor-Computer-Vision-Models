FROM python:latest

COPY main.py .
COPY haarcascade_frontalface_default.xml .
COPY yolov3.cfg .
COPY coco.names .
COPY yolov3.weights .

RUN pip install numpy fastapi opencv-python-headless urllib3 uvicorn
EXPOSE 90



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "90"]
