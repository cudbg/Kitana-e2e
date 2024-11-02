from setuptools import setup, find_packages

setup(
    name="search_engine",
    version="0.1",
    packages=find_packages(include=['search_engine', 'search_engine.*']),
    install_requires=[
        'pandas',
        'pytest',
    ],
)