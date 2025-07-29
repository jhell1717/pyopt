from setuptools import setup, find_packages

setup(
    name='pyopt',
    version='0.0.0',
    packages=find_packages(include=["pyopt"]),  # match your folders
    install_requires=[
        "torch",
        "botorch",
        "gpytorch"
    ],
)