# -*- coding:utf-8 -*-

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup


def read_requirements(file_path='requirements.txt'):
    import os

    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r')as f:
        lines = f.readlines()

    lines = [x.strip('\n').strip(' ') for x in lines]
    lines = list(filter(lambda x: len(x) > 0 and not x.startswith('#'), lines))

    return lines


def read_extra_requirements():
    import glob
    import re

    extra = {}

    for file_name in glob.glob('requirements-*.txt'):
        key = re.search('requirements-(.+).txt', file_name).group(1)
        req = read_requirements(file_name)
        if req:
            extra[key] = req

    if extra and 'all' not in extra.keys():
        extra['all'] = sorted({v for req in extra.values() for v in req})

    return extra

version = '0.1.13'
requirements = read_requirements()

MIN_PYTHON_VERSION = '>=3.6.*'

long_description = open('README.md', encoding='utf-8').read()

setup(
    name='deeptables',
    version=version,
    description='Deep-learning Toolkit for Tabular datasets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='DeepTables Community',
    author_email='yangjian@zetyun.com',
    license='Apache License 2.0',
    install_requires=requirements,
    python_requires=MIN_PYTHON_VERSION,
    extras_require={
        'tests': ['pytest', ],
        'gpu': ['tensorflow-gpu>=2.0.0', ]
    },

    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests', 'examples')),
    package_data={
        # 包含data目录下所有的.dat文件
        'deeptables': ['datasets/*.csv'],
    },
    zip_safe=False,
    include_package_data=True,
)
