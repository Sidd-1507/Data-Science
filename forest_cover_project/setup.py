from setuptools import setup, find_packages

setup(
    name="forest-cover-type-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-class classification of forest cover types using cartographic variables",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/forest-cover-type-prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
