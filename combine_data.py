from pathlib import Path

def merge_books(data_folder='data'):
    files = sorted(Path(data_folder).glob("*.txt"))
    with open(Path(data_folder) / "corpus.txt", "w", encoding="utf-8") as outfile:
        for file in files:
            if file.name != "corpus.txt":
                with open(file, encoding="utf-8") as f:
                    outfile.write(f.read() + "\n")

if __name__ == "__main__":
    merge_books()
