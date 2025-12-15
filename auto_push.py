import subprocess

def git_push():
    subprocess.run(["python3", "convert.py"], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        ["git", "commit", "-m", "Auto update pothole map"],
        check=False
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)

