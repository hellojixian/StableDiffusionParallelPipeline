# StableDiffusionParallelPipeline
this pipeline allow stable diffusion to use multi-GPU resources to speed up single image generation

In this code sample, i refactored the txt2img pipeline, but other pipelines such as img2img are similar concept.
in a standard pipeline, diffusers need to generated the text guidance latents vector and unguidance latents for generate one single images, this logic is related to CFG_Scale settings, this step requires the text_encoder and unet to work together, but actually these 2 images can be generated in parallel from different GPU resources and pull then together into one GPU to sum the result.

when you have more than one GPU, you can allow these tasks runs on in parallel on both GPUs, as the result, the speed of generating one image is almost *doubled*,

in my development environment, i have using dual RTX4090, with below sampling steps 100, and resultion 512x512, the speed improved from 6s to only 1.8s for generate a single image.

## Sample code
```python
import torch
from libs.StableDiffusionParallelPipeline import StableDiffusionParallelPipeline
from libs.benchmark import benchmark
model_path = 'runwayml/stable-diffusion-v1-5'

prompt = ["a photograph of an astronaut riding a horse"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 100            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise

pipe = StableDiffusionParallelPipeline.from_pretrained(model_path)

image = pipe(prompt, width, height, generator, num_inference_steps, guidance_scale, output).images[0]

```

## testing
```sh
# for standard version
python sd-baseline.py

# for paralleled version
python sd-parallel.py
```

## docker
### build
```sh
docker build -t sd-parallel .
```
### run
```sh
docker run --rm --name sd-parallel -v ~/.cache:/root/.cache sd-parallel
```