from setuptools import setup

setup(
    name='PetroOps',
    version = '0.0.1',
    url='https://github.com/ruckerk/PetroOps',
    author='W Kurt Rucker',
    author_email='william.rucker@gmail.com',
    packages=['PetroOps'],
    install_requires=['numpy'],
    python_requires='>3.0.0',
    license='MIT',
    description='Oilfield Operations Tools',
    long_description=open('README.txt').read()
)
