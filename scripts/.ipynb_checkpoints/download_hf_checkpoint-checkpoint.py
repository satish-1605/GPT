from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Satish1102/gpt-300m-base"
LOCAL_DIR = Path(
    "artifacts/gpt-300m-base"
)

def main():

    print("=" * 60)
    print("Downloading GPT-300M Base from Hugging Face")
    print("=" * 60)

    LOCAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(LOCAL_DIR),
    )

    print("\nDownload complete.")

    print("\nDownloaded files:")

    for file in LOCAL_DIR.iterdir():
        print(f"  {file.name}")

    print("\nLocation:")
    print(LOCAL_DIR.resolve())

    print("=" * 60)


if __name__ == "__main__":
    main()