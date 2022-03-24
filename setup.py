from setuptools import setup

setup(
    name='UWI',
    url='https://github.com/ruckerk/PetroOps',
    author='W Kurt Rucker',
    author_email='william.rucker@gmail.com',
    packages=['UWI'],
    install_requires=['numpy', 'math'],
    python_requires='>3.6.2'
    license='None',
    description='Oilfield API string manager',
    long_description=open('README.txt').read()
)
