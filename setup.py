#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'pandas',
    'torch>=2.0.1',
    'torchvision>=0.15.2',
    'tqdm',
]

test_requirements = ['pytest>=3', ]

setup(
    author="Floris De Feyter",
    author_email='floris.defeyter@kuleuven.be',
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
    description="Recognite is a library to kickstart your next PyTorch-based "
    "recognition project.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='recognite',
    name='recognite',
    packages=find_packages(include=['recognite', 'recognite.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/florisdf/recognite',
    version='0.1.0-rc0',
    zip_safe=False,
)
