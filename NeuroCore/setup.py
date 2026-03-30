from setuptools import setup, find_packages

setup(
    name='qasm',                  
    version='0.1.0',               
    description='Mini Neural Network Library ',
    author='Ali Qasm',             
    author_email='AliQasm@gmail.com',
    packages=find_packages(),     
    python_requires='>=3.8',
    install_requires=[
        'numpy',                   
    ],
)