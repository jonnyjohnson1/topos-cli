from setuptools import setup, find_packages

setup(
    name='topos',
    version='0.2.7',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'topos = topos.cli:main'
        ]
    },
)