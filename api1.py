from fastapi import FastAPI, Request
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from utils.scrn_model import SCRNModel
from datetime import datetime
import os

# 加载模型和 tokenizer
output_dir = "./data_out/scrn_in-domain"
config = AutoConfig.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(
    output_dir, model_max_length=512, padding_side="right", use_fast=False
)

# 初始化模型
if "scrn" in output_dir:
    model = SCRNModel("roberta-base", config=config)
    model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))
else:
    model = AutoModelForSequenceClassification.from_pretrained(output_dir, config=config)

model.eval()

# 初始化
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        input_text = data.get("text")
        if not input_text:
            return {"error": "Input text is required", "status": 400}

        # 对于文本进行处理
        max_length = 512
        input_tokens = tokenizer.encode(input_text, add_special_tokens=True)

        # 判断是否需要分块
        if len(input_tokens) <= max_length:
            # print("Input text length:", len(input_tokens))
            # 文本长度小于等于 512，直接处理
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1).tolist()[0]
                predicted_label = torch.argmax(outputs.logits, dim=-1).item()

            final_content_type = "AI-generated" if predicted_label == 1 else "Human-written"

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = {
                "probabilities": probabilities,
                "final_label": predicted_label,
                "final_content_type": final_content_type,
                "status": 200,
                "timestamp": now
            }
            return response
        else:
            # 文本长度大于 512，进行分块
            chunks = [
                input_tokens[i:i + max_length] for i in range(0, len(input_tokens), max_length)
            ]
            chunks = sorted(chunks, key=lambda x: len(x))

            total_probabilities = [0.0, 0.0]  # 累加每个类别的概率

            for chunk in chunks:
                chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
                inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1).tolist()[0]

                # 累加概率
                total_probabilities[0] += probabilities[0]
                total_probabilities[1] += probabilities[1]

            # 计算平均概率
            num_chunks = len(chunks)
            average_probabilities = [p / num_chunks for p in total_probabilities]
            final_label = 1 if average_probabilities[1] > average_probabilities[0] else 0
            final_content_type = "AI-generated" if final_label == 1 else "Human-written"

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = {
                "probabilities": average_probabilities,
                "final_label": final_label,
                "final_content_type": final_content_type,
                "status": 200,
                "timestamp": now
            }
            return response

    except Exception as e:
        return {"error": str(e), "status": 500}

if __name__ == "__main__":
    import uvicorn
