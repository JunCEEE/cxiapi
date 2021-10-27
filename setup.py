#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    require = requirements_file.read()
    requirements = require.split()

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest>=3',
]

setup(
    author="Juncheng E",
    author_email='juncheng.e@xfel.eu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=
    "An API and analysis toolkit for CXI dataset, including hitfinding, calibration and size distribution.",
    entry_points={
        'console_scripts': [
            'cxiapi=cxiapi.cli:main',
        ],
    },
    scripts=['bin/getFrameScores'],
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cxiapi',
    name='cxiapi',
    packages=find_packages(include=['cxiapi', 'cxiapi.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JunCEEE/cxiapi',
    version='0.1.0',
    zip_safe=False,
)
