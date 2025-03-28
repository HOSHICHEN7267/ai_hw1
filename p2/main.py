import os
import csv
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import StableDiffusion3Pipeline

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

# Load Stable Diffusion 3 model with VRAM-saving features (no slicing)
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path_sd,
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()  # keep offload only

# Load generation config for Phi-4
generation_config = GenerationConfig.from_pretrained(model_path_phi4)

# Function to process each image
def generate_snoopy_style_images(content_images_folder, output_folder, caption_csv_path, style_prompt):
    # ✅ Sorted image list
    content_images = sorted(
        [f for f in os.listdir(content_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )
    os.makedirs(output_folder, exist_ok=True)

    # Prepare CSV file for storing captions and image paths
    with open(caption_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "Generated Caption", "224x224 Image Path"])  # CSV header

        for image_name in content_images:
            try:
                print(f"🔍 Processing: {image_name}")

                # Load image
                content_image_path = os.path.join(content_images_folder, image_name)
                image = Image.open(content_image_path).convert("RGB")

                # Create improved Phi-4 prompt for short, accurate caption (face-focused)
                user_prompt = '<|user|>'
                assistant_prompt = '<|assistant|>'
                prompt = (
                    f"{user_prompt}<|image_1|>Please describe the person's appearance in this image concisely. "
                    "Focus on facial features, hairstyle, facial expression, visible accessories, and the position and orientation of the face within the image. Limit to 40 tokens.<|end|>"
                    + assistant_prompt
                )

                # Run Phi-4 to generate caption
                inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
                generate_ids = phi4_model.generate(
                    **inputs,
                    max_new_tokens=40,
                    generation_config=generation_config,
                )
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                content_description = processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                # === Structured Prompt Construction ===
                task_description = "Draw this:"
                style_description = style_prompt  # passed from outside

                full_prompt = f"{task_description} {content_description}, {style_description}"

                # Truncate if over token limit
                max_total_tokens = 77
                while True:
                    tokenized = pipe.tokenizer(full_prompt, return_tensors=None)
                    if len(tokenized["input_ids"]) <= max_total_tokens:
                        break
                    # remove one word from content_description if too long
                    content_description = " ".join(content_description.split(" ")[:-1])
                    full_prompt = f"{task_description}\nContent: {content_description}\nStyle: {style_description}"

                # Generate image from Stable Diffusion
                snoopy_image = pipe(
                    full_prompt,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                ).images[0]

                # Resize to 224x224
                resized_image = snoopy_image.resize((224, 224), Image.Resampling.LANCZOS)
                resized_image_path = os.path.join(output_folder, f"{image_name}")
                resized_image.save(resized_image_path)
                print(f"✅ Saved resized image: {resized_image_path}")

                # Write to CSV
                writer.writerow([image_name, content_description, resized_image_path])

            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")

# Set your paths
content_images_folder = './content_image'
output_folder = './output_image'
caption_csv_path = './captions.csv'

# # Run the pipeline
# generate_snoopy_style_images(content_images_folder, output_folder, caption_csv_path)
