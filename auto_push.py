import subprocess

def git_push():
    try:
        subprocess.run(
            ["python3", "convert.py"],
            check=True,
            stdin=subprocess.DEVNULL
        )

        subprocess.run(
            ["git", "add", "."],
            check=True,
            stdin=subprocess.DEVNULL
        )

        subprocess.run(
            ["git", "commit", "-m", "Auto update pothole map"],
            check=False,
            stdin=subprocess.DEVNULL
        )

        subprocess.run(
            ["git", "push", "origin", "main"],
            check=False,
            stdin=subprocess.DEVNULL
        )

        print("Git push done")

    except Exception as e:
        print("Git push error:", e)
