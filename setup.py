from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='hwr',
    version='0.1',
    description='Handwriting Recognition Resources',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BYU-Handwriting-Lab/HandwritingRecognition',
    author='BYU-Handwriting-Lab',
    keywords='BYU handwriting recognition',
    packages=find_packages(),
    package_data={
        'hwr': ['wbs/data/*']
    },
    install_requires=['tensorflow', 'pandas', 'matplotlib', 'pyyaml', 'numpy', 'editdistance', 'tqdm',
                      'word_beam_search']
)
