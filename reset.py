import os
import json

IMAGE_DIR = "images"
DATA_JSON = "potholes_data.json"
GEOJSON = "potholes.geojson"

def reset_data_safe():
    print("Resetting pothole data...")

    # 1. Clear images
    if os.path.exists(IMAGE_DIR):
        for f in os.listdir(IMAGE_DIR):
            path = os.path.join(IMAGE_DIR, f)
            if os.path.isfile(path):
                os.remove(path)
    else:
        os.makedirs(IMAGE_DIR)

    # 2. Reset JSON Lines file
    open(DATA_JSON, "w").close()

    # 3. Reset GeoJSON
    with open(GEOJSON, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": []
        }, f, indent=2)

    print("Reset completed OK")
