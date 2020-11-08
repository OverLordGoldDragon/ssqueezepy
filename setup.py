from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_test.txt") as f:
    requirements_test = f.read().splitlines()

setup(
    name="ssqueezepy",
    version="0.5.0rc1",
    description="Synchrosqueezing Toolbox ported to Python",
    install_requires=requirements,
    tests_require=requirements_test,
    packages=find_packages(),
    license="TBD",
    author="OverLordGoldDragon",
)