import json
import base64
import requests
import os
import time
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import re
import cv2
import glob
import math


def gptv_query(transcript=None, top_p=0.2, temp=0., model_type="lxl-gpt-4.1", port=8000, seed=123):
    max_tokens = 512
    wait_time = 10

    model_configs = {
        "lxl-gpt-4.1": {
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer sk-XQNqYI9MNOODopXA1e23F4Ff96B045B295A5EeFe699f0eC2"
            },
            "requests_url": "http://111.170.172.25:8088/v1/chat/completions",
            "model_type": "gpt-4.1"
        },
    }

    if model_type in model_configs:
        config = model_configs[model_type]
        headers = config["headers"]
        requests_url = config["requests_url"]
        model_type = config["model_type"]

    data = {
        'model': model_type,
        'max_tokens': max_tokens,
        'temperature': temp,
        'top_p': top_p,
        'messages': [],
        'seed': seed,
    }
    if transcript is not None:
        data['messages'] = transcript
    # print("data", data)
    response_text, retry, response_json = '', 0, None
    while len(response_text) < 2:
        retry += 1
        # print(f"retry: {retry}")
        try:
            response = requests.post(url=requests_url, headers=headers, data=json.dumps(data), verify=False)
            # print("Status code:", response.status_code)
            # print("Response text:", response.text)
            response_json = response.json()
        except Exception as e:
            print(f"Exception: {e}, retrying...")
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            print(f"HTTP Status: {response.status_code}, content: {response.content}")
            time.sleep(wait_time)
            data['temperature'] = min(data['temperature'] + 0.2, 1.0)
            continue
        if 'choices' not in response_json:
            print(f"Invalid response, retrying...")
            time.sleep(wait_time)
            continue
        response_text = response_json["choices"][0]["message"]["content"]
    return response_text

# === 1. 读取评估模板 ===
with open('evaluation/character.txt', 'r') as f:
    prompt_character = f.read()
with open('evaluation/causal.txt', 'r') as f:
    prompt_causal = f.read()
with open('evaluation/emotion.txt', 'r') as f:
    prompt_emotion = f.read()
with open('evaluation/interest.txt', 'r') as f:
    prompt_interest = f.read()
with open('evaluation/narrative.txt', 'r') as f:
    prompt_narrative = f.read()
with open('evaluation/natural.txt', 'r') as f:
    prompt_natural = f.read()

eval_prompt_map = {
    "character_consistency": prompt_character,
    "causal_temporal_coherence": prompt_causal,
    "emotional_expression": prompt_emotion,
    "narrative_integrity": prompt_narrative,
    "language_naturalness": prompt_natural,
    "interest": prompt_interest,
}

# === 3. 构建 prompt ===
def load_text(text_prompt):
    text_dict = {
        "type": "text",
        "text": text_prompt,
    }
    return text_dict


def evaluate_story(txt_path, eval_prompt_map):
    from pathlib import Path
    path = Path(txt_path)
    stem = path.stem
    id = stem.split('_')[-2]

    with open(txt_path, "r", encoding='utf-8') as f:
        story_txt = f.read()

    result = {
        "text": story_txt,
    }

    for eval_type, prompt_template in eval_prompt_map.items():
        print(f"[{stem}] calculating {eval_type}...")

        transcript = [{"role": "system", "content": ""}, {"role": "user", "content": []}]
        transcript[-1]["content"].append(load_text(prompt_template))
        transcript[-1]["content"].append(load_text(story_txt))

        temp_start = 0.0
        max_retry = 5
        score = None
        analysis = ""

        while True:
            try:
                response = gptv_query(transcript=transcript, temp=temp_start)
                print(f"[{stem}] [{eval_type}] Raw Response:\n{response}")

                for line in response.strip().split("\n"):
                    if line.lower().startswith("Feedback:"):
                        analysis = line.split(":", 1)[1].strip()

                matches = re.findall(r"(score|Score):\s*[a-zA-Z]*\s*(\d+)", response)
                if len(matches) == 0:
                    raise ValueError("No valid score found")
                score = int(matches[0][1])
                break
            except Exception as e:
                temp_start += 0.1
                max_retry -= 1
                print(f"[Retry] {eval_type} failed for {stem}. Left: {max_retry}. Error: {e}")
                if max_retry == 0:
                    score = 0
                    analysis = "[Fallback] Could not extract analysis or score."
                    break

        result[eval_type] = {
            "score": score,
            "analysis": analysis
        }
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/analysis_story{id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    print(f"[{stem}] Analysis saved to {out_path}")
    return out_path

txt_dir = './subtitles_story'
out_dir = 'eva_score/eva_story'
story_files = [
    os.path.join(txt_dir, f) for f in sorted(os.listdir(txt_dir))
    if f.endswith(".txt") and 'idx' not in f
]

max_workers = 10  # 你可以改为 8 或更多，根据机器和接口限制

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(evaluate_story, story_file, eval_prompt_map): story_file
        for story_file in story_files
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating stories"):
        try:
            result_path = future.result()
        except Exception as e:
            print(f"[Error] A story failed: {e}")



