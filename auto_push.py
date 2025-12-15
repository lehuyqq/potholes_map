import subprocess

def git_push():
    try:
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
        subprocess.run(["python3", "convert.py"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(
            ["git", "commit", "-m", "Auto update pothole data"],
            check=False
        )
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Da push thanh cong")
    except Exception as e:
        print("Da xay ra loi:", e)
