from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="pyMassEvac",
    version="0.0.1",
    description="A custom gymnasium environment for studying multi-domain mass evacuation operations.",
    long_description=readme,
    long_description_content_type="text/x-rst",
    url="",
    author="Mark Rempel",
    author_email="mark.rempel@forces.gc.ca",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",   
        "Operating System :: OS Independent",  
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering"
    ],
    keywords="mass evacuation, reinforcement learning, sequential decision problem",
    python_requires=">=3.7",
    packages=find_packages(include = \
                           ["gym_mass_evacuation"], \
                            exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "gymnasium",
        "stable-baselines3",
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
