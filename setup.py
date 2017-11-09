from distutils.core import setup
from setuptools import find_packages

requirements = []
with open('requirements.txt') as req_file:
    requirements = req_file.read().splitlines()

long_description = """
ebmtoolbox
"""
setup(
    name='ebmtoolbox',
    packages=find_packages(),
    version='1.0.0-a1',
    description='ebm toolbox',
    long_description=long_description,
    author='Erasmus MC Rotterdam',
    author_email='v.venkatraghavan@erasmusmc.nl',
    maintainer="Erasmus MC Rotterdam",
    maintainer_email="v.venkatraghavan@erasmusmc.nl",
    url='https://github.com/88vikram/pyEBM',
    install_requires=['requests'],
    setup_requires=requirements
)
