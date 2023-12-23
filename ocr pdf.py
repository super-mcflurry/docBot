import io
import json
import cv2
import numpy as np
import requests
import os
from pdf2image import convert_from_path
from PIL import Image


# img_path = r"C:\Users\lians\Downloads\1922_bottled_Coca-Cola_ad.png"
# img = cv2.imread(img_path)

#for pdf
img = convert_from_path(r"C:\Users\lians\Downloads\test.pdf", 500, poppler_path=r'C:\Program Files\poppler-23.11.0\Library\bin')
first_image = img[0]
img_bytes_io = io.BytesIO()
first_image.save(img_bytes_io, format='JPEG')
img_bytes = img_bytes_io.getvalue()


# file_size_MB = os.path.getsize(img_path) / (1024 * 1024)

# if file_size_MB > 1:
#     postCompressedPercentage = 1 / file_size_MB
#     _, img = cv2.imencode(".jpg", img, [1, postCompressedPercentage * 100])
# else: 
#     _, img = cv2.imencode(".jpg", img, [1, 100])


url_api = "https://api.ocr.space/parse/image"

# #convert to bytes
# file_bytes = io.BytesIO(img)

result = requests.post(url_api,
                files = {"photo.jpg": img_bytes},
                data = {"apikey": "K88247282988957",
                        "language": "eng"})

result = result.content.decode()
result = json.loads(result)

parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")

print(text_detected)