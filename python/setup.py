from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamarag",
    version="0.1.0",
    author="LlamaSearch.AI",
    author_email="info@llamasearch.ai",
    description="A Python package for Retrieval-Augmented Generation with LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearch/llamarag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "nltk>=3.6.0",
        "fastapi>=0.68.0",
        "pydantic>=1.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
        "anthropic": [
            "anthropic>=0.5.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
        "huggingface": [
            "sentence-transformers>=2.2.0",
            "torch>=1.10.0",
        ],
        "llamadb": [
            "llamadb>=0.1.0",
        ],
        "api": [
            "uvicorn>=0.15.0",
        ],
    },
) 