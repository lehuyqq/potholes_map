import subprocess

def git_push():
    subprocess.run(["git", "add", "potholes_data.json", "potholes.geojson", "images"], check=True)
    subprocess.run(
        ["git", "commit", "-m", "Auto update pothole data"],
        check=False
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)


