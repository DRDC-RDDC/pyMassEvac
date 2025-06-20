from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="pyMassEvac",
    version="1.0.0",
    description="A custom gymnasium environment for studying single- and multi-domain mass evacuation operations.",
    long_description=readme,
    long_description_content_type="text/x-rst",
    url="",
    author="Mark Rempel",
    author_email="mark.rempel@forces.gc.ca",
    license="BSD-3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
        "License :: OSI Approved :: BSD-3 License",   
        "Operating System :: OS Independent",  
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering"
    ],
    keywords="mass evacuation, reinforcement learning, sequential decision problem",
    python_requires=">=3.7",
    packages=find_packages(include = \
                           ["pyMassEvac"], \
                            exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "gymnasium",
        "tdqm",
        "seaborn"
    ],
    extras_require={
        "dev": [
            "pip",
            "Sphinx"
        ]
    }
)
