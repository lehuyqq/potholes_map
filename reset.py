#!/usr/bin/env python3
import os
import json

IMAGE_DIR = "images"
DATA_JSON = "potholes_data.json"
GEOJSON = "potholes.geojson"

def reset_data():
    print("=== Xoa DATA ===")

    # 1. Ensure images dir
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 2. Delete all images
    for f in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, f)
        if os.path.isfile(path):
            os.remove(path)

    # 3. Reset potholes_data.json (JSON Lines)
    open(DATA_JSON, "w").close()

    # 4. Reset potholes.geojson
    with open(GEOJSON, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": []
        }, f, indent=2)

    print("Da xoa xong.")

if __name__ == "__main__":
    reset_data()
