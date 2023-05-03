import torch
from diffusers import StableDiffusionPipeline
from libs.benchmark import benchmark
model_path = 'runwayml/stable-diffusion-v1-5'

prompt = ["a photograph of an astronaut riding a horse"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 100            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
output = 'baseline'
test_loops = 5

pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe = pipe.to("cuda")

benchmark(test_loops, pipe, prompt, width, height, generator, num_inference_steps, guidance_scale, output)
