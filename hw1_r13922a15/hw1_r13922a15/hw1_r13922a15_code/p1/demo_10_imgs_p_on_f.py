import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
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

# Phi-4 on Flickr30k 獨立測試函數
def evaluate_phi4_on_flickr30k(max_images=10):
    print(f"\n===== Evaluating Phi-4 on Flickr30k =====")

    # 載入 Phi-4 模型
    model, processor, generation_config = load_phi4_model()

    # 載入 Flickr30k 數據集
    dataset = load_dataset("nlphuji/flickr30k")["test"]
    print(f"Dataset size: {len(dataset)}")  # 打印數據集大小

    # 設定預設的 `MAX_IMAGES`
    max_images = max_images if max_images is not None else 10

    # 儲存評估結果
    all_generated_captions = []
    all_reference_captions = []

    # 儲存圖片和 caption 的目錄
    os.makedirs("outputs/phi4/flickr30k", exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(min(max_images, len(dataset))), desc=f"Processing Phi-4 on Flickr30k"):

            # **處理圖片**
            data = dataset[i]
            img_data = data["image"]
            raw_image = None

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

            if not raw_image:
                print(f"未知的圖片格式，跳過")
                continue

            # **模型推理**
            prompt = '<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>'
            inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(device)
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=512,  # 設定最大新tokens為512
                generation_config=generation_config,
            )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            generated_caption = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 取得 Ground Truth captions
            reference_captions = data["caption"]

            # 儲存圖片與生成的 caption
            raw_image.save(f"outputs/phi4/flickr30k/image_{i+1}.png")
            with open(f"outputs/phi4/flickr30k/caption_{i+1}.txt", "w") as f:
                f.write(f"Generated Caption: {generated_caption}\n")
                f.write(f"Reference Caption: {reference_captions}\n")

            # 收集生成的 caption 和 Ground Truth
            all_generated_captions.append(generated_caption)
            all_reference_captions.append(reference_captions)

    # **一次性計算所有圖片的 BLEU, ROUGE, METEOR**
    bleu_result = bleu.compute(predictions=all_generated_captions, references=all_reference_captions)
    rouge_result = rouge.compute(predictions=all_generated_captions, references=all_reference_captions)
    meteor_result = meteor.compute(predictions=all_generated_captions, references=all_reference_captions)

    # **輸出結果並儲存至檔案**
    evaluation_results = f"\n===== Phi-4 on Flickr30k 評估結果 =====\n"
    evaluation_results += f"BLEU Score: {bleu_result['bleu']:.4f}\n"
    evaluation_results += f"ROUGE-1: {rouge_result['rouge1']:.4f}, ROUGE-2: {rouge_result['rouge2']:.4f}\n"
    evaluation_results += f"METEOR Score: {meteor_result['meteor']:.4f}\n"

    # 輸出到控制台
    print(evaluation_results)

    # 儲存結果到檔案
    with open("evaluation_results_phi-4_flickr30k.txt", "a") as f:
        f.write(evaluation_results)

# **執行 Phi-4 on Flickr30k**
evaluate_phi4_on_flickr30k(max_images=10)  # 預設處理 10 張圖
