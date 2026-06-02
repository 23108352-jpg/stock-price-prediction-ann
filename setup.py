from setuptools import setup, find_packages

setup(
    name="stock-ann-prediction",
    version="1.0.0",
    author="Muhammad Mubeen Khan",
    author_email="23108352@szabist-isb.edu.pk",
    description="Stock Market Trend Prediction Using Artificial Neural Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/stock-ann-prediction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tensorflow",
        "scikit-learn",
        "yfinance",
        "streamlit",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "stock-train=train:main",
            "stock-evaluate=evaluate:main",
            "stock-predict=predict:main",
        ],
    },
)
