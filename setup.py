import os
import re
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return reader.read()


def get_requirements(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                version_file, re.M)
    if version_matched:
        return version_matched.group(1)
    raise RuntimeError('Unable to find version')


setup(
    name="ssqueezepy",
    version=find_version('ssqueezepy', '__init__.py'),
    packages=find_packages(exclude=['tests', 'examples']),
    url="https://github.com/OverLordGoldDragon/ssqueezepy",
    license="MIT",
    author="OverLordGoldDragon",
    author_email="16495490+OverLordGoldDragon@users.noreply.github.com",
    description=("Synchrosqueezing and wavelet transforms in Python"),
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    keywords=(
        "signal-processing python synchrosqueezing wavelet-transform"
    ),
    install_requires=get_requirements('requirements.txt'),
    tests_require=["pytest>=4.0", "pytest-cov"],
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
