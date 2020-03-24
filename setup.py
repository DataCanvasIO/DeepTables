from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup

version = '0.1.4'

requirements = [
    'scipy>=1.3.1',
    'pandas>=0.25.3',
    'scikit-learn==0.22.1',
    'tensorflow==2.0.0',
    'numpy>=1.17.4',
    'catboost==0.20.2',
    'lightgbm==2.3.0',
    'scikit-optimize==0.7.1',
    'category_encoders==2.1.0',
]

MIN_PYTHON_VERSION = '>=3.6.*'

long_description = open('README.md').read()

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
        'gpu': ['tensorflow-gpu==2.0.0', ]
    },

    classifiers=(
        "Operating System :: OS Independent",
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
    ),
    packages=find_packages(exclude=('tests',)),
    package_data={
        # 包含data目录下所有的.dat文件
        'deeptables': ['datasets/*.csv'],
    },
    zip_safe=True,
    include_package_data=True,
)
