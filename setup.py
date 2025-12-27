# setup.py
from setuptools import setup, find_packages

setup(
    name="tigsumm",
    version="0.1.0",
    description="TigSumm: Cross-Lingual Sentiment-Aware Summarization framework",
    author="Your Name",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "torch>=1.12",
        "transformers>=4.40.0",
        "datasets",
        "sentencepiece",
        "peft",
        "accelerate",
        "evaluate",
        "scikit-learn",
        "tqdm",
    ],
    python_requires=">=3.8",
)
