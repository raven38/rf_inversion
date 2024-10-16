import torch

from diffusers import FluxImg2ImgPipeline
from pipeline_flux_rf_inversion import FluxRFInversionPipeline
from diffusers.utils import load_image

device = "cuda"
pipe = FluxRFInversionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

images = pipe(
    prompt=prompt, prompt_2=prompt, image=init_image, num_inference_steps=28, strength=0.95, guidance_scale=3.5, gamma=0.5, eta=0.9
).images[0]

images.save("output.jpg")
