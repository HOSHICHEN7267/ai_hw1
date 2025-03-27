import os
import csv
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

# === Load Phi-4 ===
phi4_model_path = "microsoft/Phi-4-multimodal-instruct"
phi4_model = AutoModelForCausalLM.from_pretrained(
    phi4_model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
).cuda()
processor = AutoProcessor.from_pretrained(phi4_model_path, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(phi4_model_path)

# === Load Stable Diffusion v1.5 for img2img ===
sd_pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
sd_pipeline.enable_model_cpu_offload()
sd_pipeline.enable_xformers_memory_efficient_attention()

# === Config ===
content_images_folder = './content_image'
output_folder = './output_image_img2img'
description_csv_path = './captions_img2img.csv'
style_prompt = "black and white Snoopy comic style, simplified lines, expressive faces"

os.makedirs(output_folder, exist_ok=True)

# === Helper: Describe image with Phi-4 ===
def describe_image_with_phi4(image: Image.Image) -> str:
    prompt = (
        "<|user|><|image_1|>Please describe the person's appearance in this image concisely. "
        "Focus on facial features, hairstyle, facial expression, visible accessories, and the position and orientation of the face within the image. Limit to 40 tokens.<|end|><|assistant|>"
    )
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
    generate_ids = phi4_model.generate(
        **inputs,
        max_new_tokens=40,
        generation_config=generation_config
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
    return response

# === Main execution ===
def main():
    content_images = sorted(
        [f for f in os.listdir(content_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )

    with open(description_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "Generated Caption", "Output Path"])

        for image_name in content_images:
            try:
                print(f"🔍 Processing: {image_name}")

                image_path = os.path.join(content_images_folder, image_name)
                pil_image = Image.open(image_path).convert("RGB")

                # === Step 1 & 2: Describe image using Phi-4 ===
                description = describe_image_with_phi4(pil_image)

                # === Step 3: Compose prompt ===
                full_prompt = f"Draw this person: {description}. Style: {style_prompt}. Do not generate any NSFW content."

                # === Step 4: Resize image to 512x512 and feed into SD1.5 img2img ===
                resized_input = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
                output_image = sd_pipeline(full_prompt, image=resized_input).images[0]

                # === Save output ===
                out_path = os.path.join(output_folder, image_name)
                output_image.save(out_path)
                print(f"✅ Saved: {out_path}")

                writer.writerow([image_name, description, out_path])

            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")

if __name__ == '__main__':
    main()
