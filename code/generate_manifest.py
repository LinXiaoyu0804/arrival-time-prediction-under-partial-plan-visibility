"""Write a deterministic SHA-256 manifest for the repository release."""

from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_MANIFEST.sha256"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and ".git" not in path.relative_to(ROOT).parts
        and "build" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.relative_to(ROOT).parts
    )
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"{len(files)} files -> {OUTPUT}")


if __name__ == "__main__":
    main()
