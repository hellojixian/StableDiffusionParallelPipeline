import torch
from PIL import Image
from timeit import default_timer as timer
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

class StableDiffusionParallelPipeline:
  def from_pretrained(model_path, torch_dtype=torch.float16):
    model = StableDiffusionParallelPipeline(model_path, torch_dtype)
    return model

  def to(self, device):
    self.vae.to(device)

  def __init__(self, model_path, torch_dtype=torch.float16):
    self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype)
    self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=torch_dtype)
    self.scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype)
    self.output_device = "cuda:0"
    self.vae.to(self.output_device)
    self.dtype = torch_dtype
    text_encoders = []
    unets = []
    for gpu_id in range(torch.cuda.device_count()):
      torch_device = f"cuda:{gpu_id}"
      text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
      text_encoders.append(text_encoder.to(torch_device))
      unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype)
      unets.append(unet.to(torch_device))

    self.text_encoders = text_encoders
    self.unets = unets

  def embed_text(self, text, gpu_id):
    text_input = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
      device =  self.text_encoders[gpu_id].device
      text_embeddings =  self.text_encoders[gpu_id](text_input.input_ids.to(device))[0]
    return text_embeddings

  def unet_pred(self, latent_model_input, t, text_embeddings, gpu_id):
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    with torch.no_grad():
      noise_pred = self.unets[gpu_id](latent_model_input.to(self.unets[gpu_id].device), t, encoder_hidden_states=text_embeddings).sample
    return noise_pred

  @torch.no_grad()
  def __call__(
      self,
      prompt, width, height, generator, num_inference_steps, guidance_scale):
    batch_size = 1
    latents = torch.randn(
      (batch_size, self.unets[0].config.in_channels, height // 8, width // 8),
      generator=generator,
    ).to(self.dtype)
    self.scheduler.set_timesteps(num_inference_steps)
    latents = latents * self.scheduler.init_noise_sigma

    inputs = ["", prompt]
    with ThreadPoolExecutor(max_workers=2) as executor:
      future_embeddings = [executor.submit(self.embed_text, inputs[i], i) for i in range(len(inputs))]
    text_embeddings = [f.result() for f in future_embeddings]

    for t in tqdm(self.scheduler.timesteps):
      with ThreadPoolExecutor(max_workers=2) as executor:
        future_noises = [executor.submit(self.unet_pred, torch.cat([latents] * 1), t, text_embeddings[i], i) for i in range(len(text_embeddings))]

      [noise_pred_uncond, noise_pred_text] = [f.result().to(self.output_device) for f in future_noises]
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
      # compute the previous noisy sample x_t -> x_t-1
      latents = self.scheduler.step(noise_pred, t, latents.to(self.output_device)).prev_sample

    latents = 1 / 0.18215 * latents

    with torch.no_grad():
      image = self.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    output = StableDiffusionParallelPipelineOutput()
    output.images = pil_images
    return output

class StableDiffusionParallelPipelineOutput:
  images = []