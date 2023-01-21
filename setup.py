# -*- coding:utf-8 -*-

from __future__ import absolute_import

import os

from setuptools import find_packages
from setuptools import setup

# try:
#     import tensorflow
#
#     tf_installed = True
# except ImportError:
#     tf_installed = False


def read_requirements(file_path='requirements.txt'):
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


def read_version(py_file, var_name='__version__'):
    try:
        execfile
    except NameError:
        def execfile(fname, globs, locs=None):
            locs = locs or globs
            exec(compile(open(fname).read(), fname, "exec"), globs, locs)

    version_ns = {}

    base_dir = os.path.dirname((os.path.abspath(__file__)))
    execfile(os.path.join(base_dir, py_file), version_ns)
    version = version_ns[var_name]

    return version


name = 'deeptables'
version = read_version(os.path.join(name, '_version.py'))
requirements = read_requirements()
# if not tf_installed:
#     requirements = ['tensorflow>=2.0.0,<2.5.0', ] + requirements

MIN_PYTHON_VERSION = '>=3.6.*'

long_description = open('README.md', encoding='utf-8').read()

setup(
    name=name,
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
    },

    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests', 'examples')),
    package_data={
        # 包含examples目录下所有文件
        'deeptables': ['examples/*'],
    },
    zip_safe=False,
    include_package_data=True,
)
