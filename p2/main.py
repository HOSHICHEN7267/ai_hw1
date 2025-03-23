import os
import csv
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

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
def generate_snoopy_style_images(content_images_folder, output_folder, caption_csv_path):
    content_images = os.listdir(content_images_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Prepare CSV file for storing captions and image paths
    with open(caption_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "Generated Caption", "224x224 Image Path"])  # CSV header

        # Use tqdm to create a progress bar for the loop
        for image_name in tqdm(content_images, desc="Processing images", unit="image"):
            try:
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

                # Add Snoopy style to prompt
                snoopy_prompt = response + ", in Snoopy comic style"

                # Use the generated text prompt for Stable Diffusion
                snoopy_image = pipe(
                    snoopy_prompt,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                ).images[0]

                # Resize image to 224x224
                resized_image = snoopy_image.resize((224, 224), Image.Resampling.LANCZOS)
                resized_image_path = os.path.join(output_folder, f"snoopy_{image_name}")
                resized_image.save(resized_image_path)
                print(f'Saved resized 224x224 image to {resized_image_path}')

                # Save info to CSV
                writer.writerow([image_name, response, resized_image_path])

            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")

# Define paths for input images and output folder
content_images_folder = './content_image'
output_folder = './output_image'  # Will contain only resized images
caption_csv_path = './captions.csv'

# Run the pipeline
generate_snoopy_style_images(content_images_folder, output_folder, caption_csv_path)
