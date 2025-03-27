# WhatIfMirror

**WhatIfMirror** is a customized pipeline for real-time image generation, built upon the foundation of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion).  
This project is part of an artistic exploration of neural diffusion technologies, tailored for interactive streams, audiovisual applications, and generative installations.

## Overview

WhatIfMirror extends and adapts existing components to enable:

- Image generation using updated diffusers *Stable Diffusion* pipelines
- Newer Pythorch version support  
- Support for *ControlNet* and *PEFT*  
- Optional acceleration (WIP)

The goal is not only technical efficiency, but also to investigate how these tools can support artistic, performative, and narrative experiences.

## Installation

Requirements:

- Python >= 3.10  
- CUDA-compatible GPU
- PyTorch

```bash
git clone https://github.com/thewhatifproject/WhatIfMirror.git
cd WhatIfMirror
pip install -e .
