import json

INPUT_FILE = "potholes_data.json"
OUTPUT_FILE = "potholes.geojson"

features = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        d = json.loads(line)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [d["lon"], d["lat"]]  # GeoJSON: lon, lat
            },
            "properties": {
                "id": d["id"],
                "confidence": d["confidence"],
                "image": d["image"],
                "time": d["time"]
            }
        }
        features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(geojson, f, indent=2)

print("GeoJSON converted")
