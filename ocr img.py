import io
import json
import cv2
import numpy as np
import requests
import os


img_path = r"C:\Users\lians\Downloads\test2.jpg"
img = cv2.imread(img_path)


_, img = cv2.imencode(".jpg", img, [1, 100])


url_api = "https://api.ocr.space/parse/image"

#convert to bytes
file_bytes = io.BytesIO(img)

result = requests.post(url_api,
                files = {"photo.jpg": file_bytes},
                data = {"apikey": "K88247282988957",
                        "language": "eng"})

result = result.content.decode()
result = json.loads(result)

parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")

print(text_detected)