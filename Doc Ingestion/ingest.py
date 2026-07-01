from pathlib import Path

# Stores the path to the "docs" folder
docs_folder = Path("Doc Ingestion/docs")

# Check if the "docs" folder exists and print its contents
print("Current working directory:", Path.cwd())
print("Docs folder exists:", docs_folder.exists())
print("Contents of docs folder:", list(docs_folder.iterdir()) if docs_folder.exists() else "Folder not found")

# Print the names and contents of all .txt files in the "docs" folder
for file in docs_folder.glob("*.txt"):
    print(file.name)

    text = file.read_text(encoding="utf-8")

    print(text)

    print("-" * 50)