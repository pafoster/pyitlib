import os
from setuptools import setup


def source_root_dir():
    """Return the path to the root of the source distribution"""
    return os.path.abspath(os.path.dirname(__file__))


def read_version():
    """Read the version from the ``pyitlib.version`` module"""
    filename = os.path.join(source_root_dir(), 'pyitlib/pyitlib_version.py')
    with open(filename) as fin:
        namespace = {}
        exec(fin.read(), namespace)  # pylint: disable=exec-used
        return namespace['__version__']


setup(
    name='pyitlib',
    version=read_version(),
    description='A library of information-theoretic methods',
    long_description=open('README.rst').read(),
    url='https://github.com/pafoster/pyitlib',
    download_url='https://github.com/pafoster/pyitlib/archive/0.2.3.tar.gz',
    author='Peter Foster',
    author_email='pyitlib@gmx.us',
    license='MIT',
    packages=['pyitlib', ],
    zip_safe=False,
    install_requires=[
        'pandas>=0.20.2',
        'numpy>=1.9.2',
        'scikit-learn>=0.16.0,<=0.24',
        'scipy>=1.0.1',
        'future>=0.16.0'
    ],
    keywords=['entropy', 'information theory', 'Shannon information',
              'uncertainty', 'correlation', 'statistics',
              'machine learning', 'data analysis', 'data science'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
