import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import evaluate
from tqdm import tqdm  # 引入 tqdm 用來顯示進度條

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
def evaluate_phi4_on_flickr30k(max_images=None):
    print(f"\n===== Evaluating Phi-4 on Flickr30k =====")

    # 載入 Phi-4 模型
    model, processor, generation_config = load_phi4_model()

    # 載入 Flickr30k 數據集
    dataset = load_dataset("nlphuji/flickr30k")["test"]

    # 設定預設的 `MAX_IMAGES`
    default_max_images = 30000  # Flickr30k 30000 張
    max_images = max_images if max_images is not None else default_max_images

    # 儲存評估結果
    all_generated_captions = []
    all_reference_captions = []

    # 使用 tqdm 顯示進度條
    for i, data in tqdm(enumerate(dataset), total=min(max_images, len(dataset)), desc="Processing Phi-4 on Flickr30k"):
        if i >= max_images:
            break  # 限制處理圖片數量

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
    with open("evaluation_results.txt", "a") as f:
        f.write(evaluation_results)

# **執行 Phi-4 on Flickr30k**
evaluate_phi4_on_flickr30k(max_images=30000)  # 可根據需要限制處理的圖片數量
