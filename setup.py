from setuptools import setup, find_packages
print find_packages()
setup(
    name="fiddler",
    version="0.1",
    packages=find_packages(),
    description="An AI that generates Irish Folk Music",
    setup_requires=[
        "numpy",
        "tensorflow",
        "click",
        "scikit-learn"
    ],
    entry_points={
        'console_scripts': ['fiddler=src.cli:main']
    }
)
