import requests, os

url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt"
save_path = "D:/Hari Project/social_moderation/weights/yolov8n-face.pt"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

r = requests.get(url, stream=True)
with open(save_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print("Downloaded yolov8n-face.pt successfully")
