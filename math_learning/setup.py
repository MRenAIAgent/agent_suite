from setuptools import setup, find_packages

setup(
    name="math_learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "math-learning=math_learning.__main__:main",
        ],
    },
) 