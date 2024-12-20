import gradio as gr
import torch
from pipeline_flux_rf_inversion import FluxRFInversionPipeline
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
import requests
from io import BytesIO
import os
import uuid
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login
import json

UPLOAD_DIR = "uploaded_images"
CONFIG_FILE = "hf_config.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_token(token):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({'token': token}, f)

def load_token():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('token')
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def authenticate_huggingface(token):
    try:
        login(token=token)
        save_token(token)  
        return "Successfully authenticated with HuggingFace!"
    except Exception as e:
        return f"Authentication failed: {str(e)}"

def cleanup_image(image_path):
    try:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        print(f"Error cleaning up image {image_path}: {e}")

def save_uploaded_image(image):
    if image is None:
        return None
    
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(filepath)
    else:
        image.save(filepath)
    
    return filepath

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

async def generate_single_image(pipe, init_image, prompt, prompt_2, num_inference_steps, strength, guidance_scale, kwargs):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                image=init_image,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                **kwargs,
            ).images[0]
        )
    return result

async def process_image_async(
    input_image,
    model_name,
    prompt,
    prompt_2,
    num_inference_steps,
    strength,
    guidance_scale,
    gamma,
    eta,
    start_timestep,
    stop_timestep,
    use_img2img,
    num_images,
    cleanup_uploads,
    
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_img2img:
        pipe = FluxImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        pipe = FluxRFInversionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    pipe = pipe.to(device)

    image_path = save_uploaded_image(input_image)
    
    init_image = Image.open(image_path).convert('RGB')
    init_image = init_image.resize((1024, 1024))
    
    prompt_2 = prompt_2 if prompt_2 else prompt
    
    kwargs = {"gamma": gamma, "eta": eta, "start_timestep": start_timestep, "stop_timestep": stop_timestep} if not use_img2img else {}
    
    tasks = [
        generate_single_image(
            pipe, init_image, prompt, prompt_2, 
            num_inference_steps, strength, guidance_scale, kwargs
        )
        for _ in range(num_images)
    ]
    
    output_images = await asyncio.gather(*tasks)
    
    if cleanup_uploads:
        cleanup_image(image_path)
    
    return output_images

def process_image(
    input_image,
    model_name,
    prompt,
    prompt_2,
    num_inference_steps,
    strength,
    guidance_scale,
    gamma,
    eta,
    start_timestep,
    stop_timestep,
    use_img2img,
    num_images,
    cleanup_uploads
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            process_image_async(
                input_image,
                model_name,
                prompt,
                prompt_2,
                num_inference_steps,
                strength,
                guidance_scale,
                gamma,
                eta,
                start_timestep,
                stop_timestep,
                use_img2img,
                num_images,
                cleanup_uploads
            )
        )
    finally:
        loop.close()


with gr.Blocks() as demo:
    gr.Markdown("# Flux RF Inversion Pipeline")
    with gr.Accordion( label = 'HF_TOKEN (for dev)', open=False):
        with gr.Row():
            with gr.Column():
                token_input = gr.Textbox(
                    label="HuggingFace Token", 
                    type="password",
                    value=load_token(), 
                    placeholder="Enter your HuggingFace token here"
                )
                auth_button = gr.Button("Authenticate")
                auth_status = gr.Textbox(label="Authentication Status", interactive=False)
                
                auth_button.click(
                    fn=authenticate_huggingface,
                    inputs=[token_input],
                    outputs=[auth_status]
                )
        
    gr.Markdown("---")  
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            model_name = gr.Text(
                value="black-forest-labs/FLUX.1-schnell",
                label="Model Name"
            )
            prompt = gr.Text(
                value="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
                label="Prompt"
            )
            prompt_2 = gr.Text(
                value="",
                label="Second Prompt (optional)"
            )
            
            with gr.Row():
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=28,
                    step=1,
                    label="Number of Inference Steps"
                )
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.01,
                    label="Strength"
                )
                
            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=3.5,
                    step=0.1,
                    label="Guidance Scale"
                )
                gamma = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Gamma"
                )
                
            with gr.Row():
                eta = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Eta"
                )
                num_images = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    label="Number of Images"
                )
                
            with gr.Row():
                start_timestep = gr.Number(
                    value=0,
                    label="Start Timestep"
                )
                stop_timestep = gr.Number(
                    value=6,
                    label="Stop Timestep"
                )
                
            with gr.Row():
                use_img2img = gr.Checkbox(
                    value=False,
                    label="Use Img2Img Pipeline"
                )
                cleanup_uploads = gr.Checkbox(
                    value=True,
                    label="Clean up uploaded images after processing"
                )
            
            generate_button = gr.Button("Generate")
        
        with gr.Column():
            output_gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=2
            )
    
    gr.Examples(
        examples=[
            [
                "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
                "black-forest-labs/FLUX.1-schnell",
                "majestic mountain landscape, snow peaks, dramatic lighting, ethereal, 8k",
                "",
                28,
                0.95,
                3.5,
                0.5,
                0.9,
                0,
                6,
                False,
                1
            ],
        ],
        inputs=[
            input_image,
            model_name,
            prompt,
            prompt_2,
            num_inference_steps,
            strength,
            guidance_scale,
            gamma,
            eta,
            start_timestep,
            stop_timestep,
            use_img2img,
            num_images,
            cleanup_uploads
        ],
    )
    
    generate_button.click(
        fn=process_image,
        inputs=[
            input_image,
            model_name,
            prompt,
            prompt_2,
            num_inference_steps,
            strength,
            guidance_scale,
            gamma,
            eta,
            start_timestep,
            stop_timestep,
            use_img2img,
            num_images,
            cleanup_uploads
        ],
        outputs=output_gallery
    )

if __name__ == "__main__":
    demo.launch()
