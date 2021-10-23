from setuptools import setup

# This will package CNNWake project as a model that can be imported
# after it is installed in the python environment using: pip install .

setup(
   name='CNNWake',
   version='1.0',
   description='CNNwake package for wake control',
   author='Jens Bauer',
   author_email='jens.bauer20@imperial.ac.uk',
   packages=['CNNWake'],
   install_requires=[
        'torch==1.9.0',
        'numpy==1.21.0',
        'scipy==1.7.0',
        'torch==1.9.0',
        'matplotlib==3.4.2',
        'FLORIS==2.4',
        'pytest==6.2.4'],
)