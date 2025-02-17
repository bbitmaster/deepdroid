from setuptools import setup, find_packages

setup(
    name="deepdroid",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.9.1",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.8.0",
        "click>=8.1.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.23.2",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "deepdroid=deepdroid.cli:cli",
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A minimal framework for building LLM-powered agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepdroid",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
) 