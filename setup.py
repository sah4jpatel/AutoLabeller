from setuptools import setup, find_packages

setup(
    name="autolabeller",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'tqdm',
        'ultralytics',
        'screeninfo',
        'pyyaml',
        'pillow',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'autolabel=AutoLabeller.main:main',
        ],
    }
)
