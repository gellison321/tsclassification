from distutils.core import setup
from setuptools import find_packages

setup(
  name = 'tsclassification',
  packages = find_packages(),
  version = '1.0.1',
  license='',
  description = 'A shapelet classifier for time series data.',
  author = 'Grant Ellison',
  author_email = 'gellison321@gmail.com',
  url = 'https://github.com/gellison321/tsclassification',
  download_url = 'https://github.com/gellison321/tsclassification/archive/refs/tags/1.0.1.tar.gz',
  keywords = ['timeseries', 'data science','data analysis', 'classificaiton', 'machine learning'],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
  install_requires=['numpy', 'sklearn', 'tsshapelet'],
)