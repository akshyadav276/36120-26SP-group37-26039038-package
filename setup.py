from setuptools import setup, find_packages

setup(
    name='nba-draft-model',
    version='1.0.0',
    description='NBA Draft Career Longevity Prediction Model',
    author='Akshita Yadav',
    author_email='Akshita.Yadav@student.uts.edu.au',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'joblib>=1.1.0',
    ],
    python_requires='>=3.8',
)