import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm  # 引入 tqdm 進度條

# Set up paths and models
model_path_phi4 = "microsoft/Phi-4-multimodal-instruct"
model_path_sd = "stabilityai/stable-diffusion-3-medium-diffusers"

# Load Phi-4 model and processor
processor = AutoProcessor.from_pretrained(model_path_phi4, trust_remote_code=True)
phi4_model = AutoModelForCausalLM.from_pretrained(
    model_path_phi4,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
).cuda()

# Load Stable Diffusion 3 model
pipe = StableDiffusion3Pipeline.from_pretrained(model_path_sd, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Load generation config for Phi-4
generation_config = GenerationConfig.from_pretrained(model_path_phi4)

# Function to process each image
def generate_snoopy_style_images(content_images_folder, output_folder):
    content_images = os.listdir(content_images_folder)
    
    # Use tqdm to create a progress bar for the loop
    for image_name in tqdm(content_images, desc="Processing images", unit="image"):
        # Prepare image and instruction for Phi-4
        content_image_path = os.path.join(content_images_folder, image_name)
        image = Image.open(content_image_path).convert("RGB")

        # Create the Phi-4 prompt for describing the content
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'

        # Feed image to Phi-4
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
        generate_ids = phi4_model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Get the generated text prompt
        print(f'Generated text prompt for image {image_name}: {response}')

        # Use the generated text prompt for Stable Diffusion
        snoopy_image = pipe(
            response,  # Generated text prompt
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]

        # Save the output image
        snoopy_image_path = os.path.join(output_folder, f"snoopy_{image_name}")
        snoopy_image.save(snoopy_image_path)
        print(f'Saved Snoopy-style image to {snoopy_image_path}')

# Define paths for input images and output folder
content_images_folder = 'path_to_your_content_images'  # Replace with your content image folder
output_folder = 'path_to_output_images'  # Replace with your desired output folder

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images with progress bar
generate_snoopy_style_images(content_images_folder, output_folder)
