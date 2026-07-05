"""Setup."""

import setuptools

DIST_NAME = "lanka_data"
VERSION = "0.1.0"
DESCRIPTION = (
    "One API to rule them all: a single interface to access "
    "public data about Sri Lanka."
)
INSTALL_REQUIRES = [
    "requests",
    "matplotlib",
    "numpy",
    "pandas",
    "geopandas",
    "shapely",
    "rich>=13.9.4",
]

setuptools.setup(
    name="lanka-data",
    version=VERSION,
    author="Nuwan I. Senaratna",
    author_email="nuuuwan@gmail.com",
    description=DESCRIPTION,
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/nuuuwan/%s" % DIST_NAME,
    project_urls={
        "Bug Tracker": "https://github.com/nuuuwan/%s/issues" % DIST_NAME,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={
        "lanka_data.datasets.dataset.custom": ["*.json"],
        "lanka_data.visual.plot": ["fonts/*.ttf"],
    },
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    test_suite="nose.collector",
    tests_require=["nose"],
)
