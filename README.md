# WhatIfMirror

**WhatIfMirror** is a customized pipeline for real-time image generation, built upon the foundation of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion).  
This project is part of an artistic exploration of neural diffusion technologies, tailored for interactive streams, audiovisual applications, and generative installations.

## Overview

WhatIfMirror extends and adapts existing components to enable:

- Image generation using optimized *Stable Diffusion* models  
- Support for *ControlNet* and *PEFT* pipelines  
- Integration with real-time tools such as *StreamDiffusion* and *AudioLDM*  
- Optional acceleration with CUDA and TensorRT  

The goal is not only technical efficiency, but also to investigate how these tools can support artistic, performative, and narrative experiences.

## Installation

Requirements:

- Python >= 3.10  
- CUDA-compatible GPU (optional, for acceleration)  
- PyTorch

```bash
git clone https://github.com/thewhatifproject/WhatIfMirror.git
cd WhatIfMirror
pip install -e .
