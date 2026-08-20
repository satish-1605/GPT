from pathlib import Path

file_path = Path("data/raw/fineweb_10k.txt")

print(f"File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

with file_path.open("r", encoding="utf-8") as file:
    print(file.read(1000))