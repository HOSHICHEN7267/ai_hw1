import os
import csv
import subprocess
from archive_main import generate_snoopy_style_images
from PIL import Image
import re

# === CONFIG ===
content_images_folder = './content_image'
style_images_folder = './style_image/snoopy'
resized_content_folder = './content_image_resized_224'
experiments_root = './experiments'
caption_csv_path = './prompt_results.csv'
# batch_size = 8
device = 'cuda'
content_metric = 'lpips'
eval_mode = 'art_fid'
eval_script = 'eval_artfid.py'
eval_script_dir = './evaluation'  # Location of eval_artfid.py and dependencies

# === 5 Recommended Prompt Variations ===
prompt_suffix_list = [
    ", in Snoopy comic style",
]

# === Resize content images to 224x224 ===
os.makedirs(resized_content_folder, exist_ok=True)
for filename in os.listdir(content_images_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(content_images_folder, filename)
        output_path = os.path.join(resized_content_folder, filename)
        try:
            img = Image.open(input_path).convert('RGB')
            img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_resized.save(output_path)
            print(f"✅ Resized: {filename}")
        except Exception as e:
            print(f"❌ Error resizing {filename}: {e}")

# === Run experiments ===
os.makedirs(experiments_root, exist_ok=True)

with open(caption_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Prompt Index', 'Prompt Suffix', 'ArtFID', 'FID', 'LPIPS', 'LPIPS_gray'])

    for idx, suffix in enumerate(prompt_suffix_list):
        print(f"\n🚀 Running experiment for prompt_{idx}: {suffix}")

        output_folder = os.path.join(experiments_root, f'prompt_{idx}')
        os.makedirs(output_folder, exist_ok=True)

        # Generate images using current prompt
        generate_snoopy_style_images(
            content_images_folder=content_images_folder,
            output_folder=output_folder,
            caption_csv_path=os.path.join(output_folder, 'captions.csv'),
            custom_suffix=suffix
        )

        # Run evaluation inside eval_script_dir
        result = subprocess.run([
            'python', eval_script,
            '--cnt', os.path.abspath(resized_content_folder),
            '--sty', os.path.abspath(style_images_folder),
            '--tar', os.path.abspath(output_folder),
            # '--batch_size', str(batch_size),
            '--device', device,
            '--mode', eval_mode,
            '--content_metric', content_metric
        ], cwd=eval_script_dir, capture_output=True, text=True)

        stdout = result.stdout
        print(stdout)
        try:
            match = re.search(r"ArtFID:\s*([\d.]+).*?FID:\s*([\d.]+).*?LPIPS:\s*([\d.]+).*?LPIPS_gray:\s*([\d.]+)", stdout)
            if match:
                numbers = [float(g) for g in match.groups()]
                writer.writerow([idx, suffix] + numbers)
            else:
                raise ValueError("ArtFID output not found or incomplete")
        except Exception as e:
            print(f"❌ Failed to parse eval results for prompt_{idx}: {e}")
            writer.writerow([idx, suffix, 'ERROR', 'ERROR', 'ERROR', 'ERROR'])
