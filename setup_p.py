"""
HARI Content Moderation System - Setup Script
Makes the project pip-installable
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="hari-content-moderation",
    version="2.0.0",
    author="HARI Project Team",
    description="Production-ready AI content moderation system with face and toxic text detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/hari-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "easyocr>=1.7.0",
        "detoxify>=0.5.0",
        "pyyaml>=6.0.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hari-demo=demo_app:main",
        ],
    },
)
