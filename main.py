import argparse
import torch
from pipeline_flux_rf_inversion import FluxRFInversionPipeline
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image

def main():
    parser = argparse.ArgumentParser(description="Run Flux RF Inversion Pipeline")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell", help="Model name or path")
    parser.add_argument("--image", type=str, default="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", help="URL of the input image")
    parser.add_argument("--prompt", type=str, default="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k", help="Prompt for image generation")
    parser.add_argument("--prompt_2", type=str, help="Second prompt (if different from prompt)")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--strength", type=float, default=0.95, help="Strength parameter")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter")
    parser.add_argument("--eta", type=float, default=0.9, help="Eta parameter")
    parser.add_argument("--start_timestep", type=int, default=0, help="Start timestep")
    parser.add_argument("--stop_timestep", type=int, default=6, help="Stop timestep")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image filename")
    parser.add_argument("--use_img2img", action="store_true", help="Use FluxImg2ImgPipeline instead of RF Inversion")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.use_img2img:
        pipe = FluxImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    else:
        pipe = FluxRFInversionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    
    pipe = pipe.to(device)

    init_image = load_image(args.image).resize((1024, 1024))
    
    prompt_2 = args.prompt_2 if args.prompt_2 else args.prompt

    images = pipe(
        prompt=args.prompt,
        prompt_2=prompt_2,
        image=init_image,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        gamma=args.gamma,
        eta=args.eta,
        start_timestep=args.start_timestep,
        stop_timestep=args.stop_timestep,
    ).images[0]

    images.save(args.output)
    print(f"Output image saved as {args.output}")

if __name__ == "__main__":
    main()