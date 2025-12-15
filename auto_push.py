import subprocess

def git_push():
    try:
        subprocess.run(["python3", "convert.py"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Auto update pothole map"], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Da push Data")
    except Exception as e:
        print("Push loi:", e)
