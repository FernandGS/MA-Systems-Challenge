import os
import pyperclip

def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(folder) if f.endswith(".py")]

    combined = ""
    for f in files:
        with open(os.path.join(folder, f), "r", encoding="utf-8") as file:
            combined += f"### {f}\n```python\n{file.read()}\n```\n\n"

    pyperclip.copy(combined)
    print(f"Copied {len(files)} .py files into clipboard.")

if __name__ == "__main__":
    main()
