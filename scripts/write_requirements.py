"""Write pinned requirement files from the packages installed in the current venv."""
import sys
from importlib import metadata

RUNTIME = ["numpy", "opencv-python", "onnxruntime"]
DEV = ["pytest", "dvc"]


def pin(names):
    lines, missing = [], []
    for n in names:
        try:
            lines.append("%s==%s" % (n, metadata.version(n)))
        except metadata.PackageNotFoundError:
            missing.append(n)
    return lines, missing


def main():
    rt, m1 = pin(RUNTIME)
    dev, m2 = pin(DEV)
    head = "# written by scripts/write_requirements.py from the working venv (Python %s)\n" % sys.version.split()[0]
    with open("requirements.txt", "w") as f:
        f.write(head + "\n".join(rt) + "\n")
    with open("requirements-dev.txt", "w") as f:
        f.write(head + "-r requirements.txt\n" + "\n".join(dev) + "\n")
    print("requirements.txt:", rt)
    print("requirements-dev.txt:", dev)
    if m1 or m2:
        print("NOT INSTALLED in this venv:", m1 + m2)


if __name__ == "__main__":
    main()

