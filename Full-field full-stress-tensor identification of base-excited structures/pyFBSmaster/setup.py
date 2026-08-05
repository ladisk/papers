#!/usr/bin/env python

import os
import re

"""The setup script."""

from setuptools import setup, find_packages


try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    try:
        from pip._internal.download import PipSession
    except ImportError:  # for pip >= 20
        from pip._internal.network.session import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession


with open('README.rst') as readme_file:
    readme = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

requirements = parse_requirements('requirements.txt', session=PipSession())

try:
    all_requirements = [str(requirement.req) for requirement in requirements]
except AttributeError:
    all_requirements = [str(requirement.requirement) for requirement in requirements]

	
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('./data')


setup(
    author="Tomaž Bregar, Ahmed El Mahmoudi, Miha Kodrič, Domen Ocepek, Francesco Trainotti, Miha Pogačar, Mert Göldeli",
    author_email='info.pyfbs@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
		'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    description="pyFBS: A Python package for Frequency Based Substructuring and Transfer Path Analysis",
    entry_points={
        'console_scripts': [
            'pyFBS=pyFBS.cli:main',
        ],
    },
    install_requires=all_requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    package_data={'': extra_files},
    keywords='pyFBS',
    name='pyFBS',
    packages=["pyFBS"],
    test_suite='tests',
    url='https://pyfbs.readthedocs.io/en/latest/intro.html',
    version='0.3.1',
)
