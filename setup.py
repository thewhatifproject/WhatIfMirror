import re

from setuptools import find_packages, setup

_deps = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "fire",
    "omegaconf",
    "pywin32;sys_platform == 'win32'",
    "controlnet-aux==0.0.9",
    "huggingface_hub",
    "numpy",
    "peft"
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

install_requires = [
    deps["fire"],
    deps["omegaconf"],
    deps["diffusers"],
    deps["transformers"],
    deps["accelerate"],
    deps["controlnet-aux"],
    deps["huggingface_hub"],
    deps["numpy"],
    deps["peft"]
]

setup(
    name="whatifmirror",
    version="0.1.0",
    description="Real-time interactive image generation pipeline based on a customized version of StreamDiffusion",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion pytorch stable diffusion audioldm streamdiffusion real-time",
    license="Apache 2.0 License",
    author="Daniele Giannini",
    author_email="contact.thewhatifproject@gmail.com",
    url="https://github.com/thewhatifproject/WhatIfMirror",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"whatifmirror": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
    ],
    project_urls={
        "Original project (StreamDiffusion)": "https://github.com/cumulo-autumn/StreamDiffusion"
    }
)