from setuptools import setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="amazon_analysis",
    version="0.0.1",
    description="mining for product sustainability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nawres AYADHI",
    zip_safe=False,
    install_requires=requirements,
)
