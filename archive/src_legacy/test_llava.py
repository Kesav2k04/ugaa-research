# src/test_llava.py — KESAV only — FINAL VERSION
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests, io

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    "D:/models/llava-1.5-7b",
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained("D:/models/llava-1.5-7b")
print("Model loaded.")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(io.BytesIO(requests.get(url).content))

question = "Is there a cat in the image?"
prompt = f"USER: <image>\n{question} Answer yes or no only.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to(model.device) if hasattr(v, "to") else v
          for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=5)

answer = processor.decode(output[0], skip_special_tokens=True)
print(f"Q: {question}")
print(f"A: {answer.split('ASSISTANT:')[-1].strip()}")
print("SUCCESS")