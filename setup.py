from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="akshita-aml-26039038",
    version="2.0.0",
    description="ML utilities from UTS 36120: NBA draft model (AT1) and Sydney weather features (AT2)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Akshita Yadav",
    author_email="Akshita.Yadav@student.uts.edu.au",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "requests>=2.28",
        "scikit-learn>=1.3",
        "joblib>=1.3",
    ],
    python_requires=">=3.10",
)