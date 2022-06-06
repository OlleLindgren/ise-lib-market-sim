"""ise-lib-market-sim setup.py"""
from pathlib import Path
import setuptools

SRC_ROOT = Path(__file__).parent
ENCODING = "UTF-8"

with open(SRC_ROOT / "README.md", "r", encoding=ENCODING) as fh:
    long_description = fh.read()

with open(SRC_ROOT / "requirements.txt", "r", encoding=ENCODING) as f:
    requirements = [r.strip() for r in f.readlines() if r.strip() if not r.startswith("git+")]
    git_requirements = [r.strip() for r in f.readlines() if r.strip() if r.startswith("git+")]

# If any git requirements, install them separately
if git_requirements:
    import sys
    import subprocess

    for requirement in git_requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", requirement], check=True)

with open(SRC_ROOT / "__init__.py", "r", encoding=ENCODING) as f:
    version_line = next(filter(lambda l: "version" in l, f.readlines()))
    version = version_line.split("=")[-1].strip(" \"'\n")

setuptools.setup(
    name="ise-lib-market-sim",
    version=version,
    author="Olle Lindgren",
    author_email="lindgrenolle@live.se",
    description="ISE I/O package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OlleLindgren/ise-lib-market-sim",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
