import os
import csv
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm

# === Load Phi-4 model and processor ===
model_path_phi4 = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_path_phi4, trust_remote_code=True)
phi4_model = AutoModelForCausalLM.from_pretrained(
    model_path_phi4,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
).cuda()
generation_config = GenerationConfig.from_pretrained(model_path_phi4)

# === Config ===
snoopy_style_folder = './style_image/snoopy'
description_output_file = './snoopy_image_descriptions.csv'
summary_output_file = './snoopy_style_summary.txt'
split_count = 5  # Number of groups to split into

# === Generate caption for each image ===
def describe_image_with_phi4(image: Image.Image) -> str:
    prompt = "<|user|><|image_1|>Describe the art style of this image in a short sentence.<|end|><|assistant|>"
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
    generate_ids = phi4_model.generate(
        **inputs,
        max_new_tokens=50,
        generation_config=generation_config
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
    return response

# === Step 1: Describe each image ===
os.makedirs(os.path.dirname(description_output_file), exist_ok=True)
all_descriptions = []
image_files = sorted([f for f in os.listdir(snoopy_style_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

with open(description_output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Image Name", "Description"])

    for filename in tqdm(image_files, desc="Describing Snoopy images"):
        path = os.path.join(snoopy_style_folder, filename)
        try:
            img = Image.open(path).convert("RGB")
            desc = describe_image_with_phi4(img)
            writer.writerow([filename, desc])
            all_descriptions.append(desc)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# === Step 2: Split descriptions and summarize each chunk ===
def summarize_descriptions(descriptions: list) -> str:
    joined = "\n".join(f"- {d}" for d in descriptions)
    prompt = (
        "<|user|>Here are style descriptions of several Snoopy-style comic images:\n\n"
        + joined +
        "\n\nPlease summarize the common visual art characteristics in a short sentence under 30 tokens."
        " This will be used as a style instruction for a text-to-image model.<|end|><|assistant|>"
    )

    inputs = processor(text=prompt, return_tensors='pt').to('cuda:0')
    generate_ids = phi4_model.generate(
        **inputs,
        max_new_tokens=50,
        generation_config=generation_config
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
    return response

chunk_size = len(all_descriptions) // split_count
summaries = []

print("\n🧠 Generating group style summaries...")
for i in range(split_count):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < split_count - 1 else len(all_descriptions)
    chunk = all_descriptions[start:end]
    print(f"📚 Summarizing group {i+1}/{split_count} with {len(chunk)} descriptions...")
    summary = summarize_descriptions(chunk)
    summaries.append(summary)

# Save all group summaries
with open(summary_output_file, 'w', encoding='utf-8') as f:
    for i, summary in enumerate(summaries):
        f.write(f"Prompt {i+1}: {summary}\n")

print("\n🎯 Final 5 style summaries:")
for i, summary in enumerate(summaries):
    print(f"Prompt {i+1}: {summary}")
