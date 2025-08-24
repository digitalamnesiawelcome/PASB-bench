import subprocess

if __name__ == "__main__":
    cmd = [
        "python", "pasb_lite.py",
        "--mode", "local",
        "--model", "gpt2",
        "--prompts", "data/prompts.json",
        "--config", "data/config.yaml"
    ]
    subprocess.run(cmd, check=True)
