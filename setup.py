from setuptools import setup

setup(
    name='UWI',
    version = '0.0.1'
    url='https://github.com/ruckerk/PetroOps',
    author='W Kurt Rucker',
    author_email='william.rucker@gmail.com',
    packages=['UWI'],
    install_requires=['numpy'],
    python_requires='>3.6.2',
    license='MIT',
    description='Oilfield API string manager',
    long_description=open('README.txt').read()
)
