"""Package current working files (including uncommitted edits) for Colab upload."""
import argparse
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("~/proj/clip-repr-colab.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
    ).decode().split("\0")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names) - {""}):
            source = root / name
            if source.is_file() and source.resolve() != output:
                archive.write(source, "clip-repr/" + name)
    print(output)


if __name__ == "__main__":
    main()
