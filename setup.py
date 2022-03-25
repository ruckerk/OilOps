from setuptools import setup

setup(
    name='PetroOps',
    version = '0.0.2',
    url='https://github.com/ruckerk/PetroOps',
    author='W Kurt Rucker',
    author_email='william.rucker@gmail.com',
    packages=['PetroOps'],
    install_requires=['numpy>0.0.0'],
    license='MIT',
    description='Oilfield Operations Tools',
    long_description=open('README.txt').read()
)
