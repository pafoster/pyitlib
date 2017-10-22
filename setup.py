from setuptools import setup
import os


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
    url='https://github.com/pafoster',
    download_url = 'https://github.com/pafoster/pyitlib/archive/0.1.10.tar.gz',
    author='Peter Foster',
    author_email='pyitlib@gmx.us',
    packages=['pyitlib', ],
    zip_safe=False,
    install_requires=[
        'pandas>=0.20.2'
        'numpy>=1.9.2',
        'scikit-learn>=0.16.0',
    ],
    keywords = ['entropy', 'information', 'uncertainty'],
)
