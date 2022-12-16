from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.6"
DESCRIPTION = "Building attention mechanisms and Transformer models from scratch. Alias ATF."

# Setting up
setup(
    name="Attention_and_Transformers",
    version=__version__,
    author="Vaibhav Singh",
    author_email="vaibhav.singh.3001@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veb-101/Attention-and-Transformers",
    # packages=find_packages(exclude=["tests"]),
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="tensorflow keras attention transformers",
    install_requires=[
        "tensorflow-macos;platform_system=='Darwin'",
        "tensorflow==2.10.0;platform_system!='Darwin'",
        "tensorflow-addons;platform_machine!='aarch64' and platform_machine!='aarch32'",
        "tensorflow-datasets",
        "livelossplot",
        "Pillow",
        "opencv-contrib-python",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scikit-image",
    ],
    python_requires=">=3.9,<3.11.*",
    license="Apache 2.0",
)
