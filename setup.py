from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

dependencies = [
    'edflow==0.4.0',
    'PyYAML',
    'matplotlib',
    'numpy',
    'pytest',
    'streamlit',
    'torch',
    'torchvision',
    'tqdm',
]

setup(
    name="VUNet",
    version="0.1.0",
    author="Johannes Haux, Patrick Esser, Andreas Blattmann, Hannes Perrot",
    author_email="jo.mobile.2011@gmail.com",
    description="A Variational Unet for person image generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jhaux/VUNet",
    include_package_data=True,
    packages=find_packages(),
    install_requires=dependencies,
    classifiers=["Programming Language :: Python :: 3",],
)
