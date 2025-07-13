from setuptools import find_packages, setup

setup(
    name="tesseraclassifier",
    packages=find_packages(where="src"),  # Find packages in the 'src' directory
    package_dir={"": "src"},
    version="0.1.0",
    description="Demo app for classifying habitats using Tessera",
    author="Jovana Knezevic",
    author_email="jovana.p.knezevic@gmail.com",
    license="MIT",
    url="https://github.com/mahuna13/tessera-classifier",
)
