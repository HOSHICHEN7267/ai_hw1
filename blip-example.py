import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import evaluate

# 檢查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入 BLIP 模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 載入 dataset
dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")
ds = dataset["test"]

# 設定最多處理的圖片數量
MAX_IMAGES = 5000

# 載入評估指標
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# 儲存評估結果
bleu_scores, rouge_scores, meteor_scores = [], [], []

for i, data in enumerate(ds):
    if i >= MAX_IMAGES:
        break  # 限制處理圖片數量

    # **檢查 image 欄位是否為 URL 或 PIL.Image**
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

    # **Unconditional image captioning**
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption_uncond = processor.decode(out[0], skip_special_tokens=True)

    # 取得 Ground Truth captions
    reference_captions = data["caption"]

    # **計算 BLEU, ROUGE, METEOR**
    bleu_result = bleu.compute(predictions=[caption_uncond], references=[reference_captions])  # 修正格式
    rouge_result = rouge.compute(predictions=[caption_uncond], references=[reference_captions])
    meteor_result = meteor.compute(predictions=[caption_uncond], references=[reference_captions])

    # 收集指標數值
    bleu_scores.append(bleu_result["bleu"])
    rouge_scores.append((rouge_result["rouge1"], rouge_result["rouge2"]))
    meteor_scores.append(meteor_result["meteor"])

    # **輸出結果**
    print(f"圖片 {i+1}: {img_data if isinstance(img_data, str) else 'PIL Image'}")
    print(f"  無條件描述: {caption_uncond}")
    print(f"  Ground Truth: {reference_captions[:3]} ...")  # 只顯示前三個
    print(f"  BLEU Score: {bleu_result['bleu']:.4f}")
    print(f"  ROUGE-1: {rouge_result['rouge1']:.4f}, ROUGE-2: {rouge_result['rouge2']:.4f}")
    print(f"  METEOR Score: {meteor_result['meteor']:.4f}")
    print("-" * 80)

# **計算所有圖片的平均指標**
if bleu_scores:
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(r[0] for r in rouge_scores) / len(rouge_scores)
    avg_rouge2 = sum(r[1] for r in rouge_scores) / len(rouge_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    print("\n===== 評估結果 (平均) =====")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}")
    print(f"METEOR Score: {avg_meteor:.4f}")
else:
    print("\n未成功處理任何圖片，請檢查資料集格式。")
