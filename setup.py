from setuptools import setup, find_packages

setup(
    name="biometric-analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if not line.startswith("#")
    ],
    python_requires=">=3.8",
) 