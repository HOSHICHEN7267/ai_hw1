import os
import csv
import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import StableDiffusion3Pipeline

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate Snoopy-style images with captions.")
parser.add_argument('--content_images_folder', type=str, required=True, help='Path to the content images folder')
parser.add_argument('--output_folder', type=str, required=True, help='Path to save the generated images')
parser.add_argument('--caption_csv_path', type=str, required=True, help='Path to save the generated captions CSV')
args = parser.parse_args()

# Fixed custom suffix (not passed in as argument)
custom_suffix = ", in Snoopy comic style"

# Load model paths
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

# Load Stable Diffusion 3 pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path_sd,
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

generation_config = GenerationConfig.from_pretrained(model_path_phi4)

# Main function
def generate_snoopy_style_images(content_images_folder, output_folder, caption_csv_path):
    content_images = sorted(
        [f for f in os.listdir(content_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )
    os.makedirs(output_folder, exist_ok=True)

    with open(caption_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "Generated Caption", "224x224 Image Path"])

        for image_name in content_images:
            try:
                print(f"🔍 Processing: {image_name}")
                content_image_path = os.path.join(content_images_folder, image_name)
                image = Image.open(content_image_path).convert("RGB")

                prompt = '<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>'
                inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

                generate_ids = phi4_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    generation_config=generation_config,
                )
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

                # Truncate to fit token limit
                max_total_tokens = 77
                while True:
                    combined_prompt = response + custom_suffix
                    tokenized = pipe.tokenizer(combined_prompt, return_tensors=None)
                    if len(tokenized["input_ids"]) <= max_total_tokens:
                        break
                    response = " ".join(response.split(" ")[:-1])

                snoopy_prompt = response + custom_suffix
                snoopy_image = pipe(
                    snoopy_prompt,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                ).images[0]

                resized_image = snoopy_image.resize((224, 224), Image.Resampling.LANCZOS)
                resized_image_path = os.path.join(output_folder, f"{image_name}")
                resized_image.save(resized_image_path)
                print(f"✅ Saved resized image: {resized_image_path}")

                writer.writerow([image_name, response, resized_image_path])

            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")

# Run main
generate_snoopy_style_images(
    args.content_images_folder,
    args.output_folder,
    args.caption_csv_path
)
