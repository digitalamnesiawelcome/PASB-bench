import os
import subprocess

if __name__ == "__main__":
    key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    cmd = [
        "python", "pasb_lite.py",
        "--mode", "api",
        "--model", "gpt-4o-mini",
        "--key", key,
        "--prompts", "data/prompts.json",
        "--config", "data/config.yaml"
    ]
    subprocess.run(cmd, check=True)
