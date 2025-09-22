# created By: Zhaoze Wang
from setuptools import setup, find_packages

setup(
    name='rtgym',
    version='1.0.0',
    description='RatatouGym: A gym environment for spatial navigation',
    author='Zhaoze Wang',
    license='MIT',
    
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.3.0',
        'scipy>=1.7.0',
        'torch>=1.10.0',
        'scikit-learn>=1.0.0',
        'faiss-cpu>=1.7.0',
        'pyyaml>=5.4.0',
        'ipython>=7.0.0',
    ],
)
