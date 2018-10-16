from setuptools import find_packages, setup

setup(
    name="segment",
    python_requires=">=3.6,<3.7",
    description="Segment",
    long_description="Segments change maps",
    author="Johannes",
    author_email="Johannes.Hansen@ed.ac.uk",
    packages=find_packages(),
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
