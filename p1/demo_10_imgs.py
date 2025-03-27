import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import evaluate
from tqdm import tqdm
import os

# 確保我們可以使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入評估指標
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# 定義 BLIP 模型
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return model, processor

# 定義 Phi-4 模型
def load_phi4_model():
    model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
    ).cuda()
    generation_config = GenerationConfig.from_pretrained(model_path)
    return model, processor, generation_config

# 通用的推理與評估函數
def evaluate_model_on_dataset(model_name, dataset_name, max_images=10):
    print(f"\n===== Evaluating {model_name} on {dataset_name} =====")

    # 載入模型
    if model_name == "BLIP":
        model, processor = load_blip_model()
    elif model_name == "Phi-4":
        model, processor, generation_config = load_phi4_model()
    else:
        raise ValueError("Invalid model name. Choose from ['BLIP', 'Phi-4']")

    # 載入數據集
    if dataset_name == "MSCOCO":
        dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")["test"]
    elif dataset_name == "Flickr30k":
        dataset = load_dataset("nlphuji/flickr30k")["test"]
    else:
        raise ValueError("Invalid dataset name. Choose from ['MSCOCO', 'Flickr30k']")

    # 儲存生成的圖片和 caption
    os.makedirs(f"outputs/{model_name}/{dataset_name}", exist_ok=True)

    # 使用 tqdm 顯示進度條
    for i, data in tqdm(enumerate(dataset), total=min(max_images, len(dataset)), desc=f"Processing {model_name} on {dataset_name}"):
        if i >= max_images:
            break

        # **處理圖片**
        img_data = data["image"]
        if isinstance(img_data, str):  # URL
            try:
                response = requests.get(img_data)
                response.raise_for_status()
                raw_image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                print(f"讀取圖片失敗 (URL): {e}")
                continue
        elif isinstance(img_data, Image.Image):  # PIL Image
            raw_image = img_data.convert("RGB")
        else:
            print(f"未知的圖片格式: {type(img_data)}，跳過")
            continue

        # **模型推理**
        if model_name == "BLIP":
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            generated_caption = processor.decode(out[0], skip_special_tokens=True)

        elif model_name == "Phi-4":
            prompt = '<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>'
            inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(device)
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                generation_config=generation_config,
            )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            generated_caption = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # 取得 Ground Truth captions
        reference_captions = data["caption"]

        # 儲存圖片與生成的 caption
        raw_image.save(f"outputs/{model_name}/{dataset_name}/image_{i+1}.png")
        with open(f"outputs/{model_name}/{dataset_name}/caption_{i+1}.txt", "w") as f:
            f.write(f"Generated Caption: {generated_caption}\n")
            f.write(f"Reference Caption: {reference_captions}\n")

    print(f"Results saved in 'outputs/{model_name}/{dataset_name}'.")

# **執行 3 種測試組合，每個處理 10 張圖**
evaluate_model_on_dataset("BLIP", "MSCOCO")  # 10 張
evaluate_model_on_dataset("BLIP", "Flickr30k")  # 10 張
evaluate_model_on_dataset("Phi-4", "MSCOCO")  # 10 張
