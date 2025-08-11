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
with open('evaluation/time_loc.txt', 'r') as f:
    prompt_time_location = f.read()
with open('evaluation/non_character.txt', 'r') as f:
    prompt_non_character = f.read()

# === 2. 图像转 base64 ===
# === 替代原 encode_image_to_base64 ===
def encode_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image to 512x512
        img = img.resize((512, 512))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")

        # Encode the image to base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_img(image_input, image_mode='path', image_input_detail='high'):
    """
    加载图片内容为API可接受的格式

    Args:
        image_input: 图片路径或URL
        image_mode: 'path' 或 'url'
        image_input_detail: 图片细节级别
    """
    if image_mode == 'url':
        return {
            "type": "image_url",
            "image_url": {
                "url": image_input,
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'path':
        base64_image = encode_image(image_input)
        image_meta = "data:image/png;base64" if 'png' in image_input else "data:image/jpeg;base64"
        return {
            "type": "image_url",
            "image_url": {
                "url": f"{image_meta},{base64_image}",
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'pil':
        img = image_input.resize((512, 512))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_meta = "data:image/png;base64"
        return {
            "type": "image_url",
            "image_url": {
                "url": f"{image_meta},{base64_image}",
                "detail": image_input_detail,
            },
        }
    else:
        raise ValueError(f"The image_mode must be either 'url' or 'path', not {image_mode}.")


# === 3. 构建 prompt ===
def load_text(text_prompt):
    text_dict = {
        "type": "text",
        "text": text_prompt,
    }
    return text_dict


json_dir = './subtitles_story'
output_dir = f"eva_score/eva_align"
os.makedirs(output_dir, exist_ok=True)
def evaluate_item(idx, item, prompt_character, prompt_time_location, prompt_non_character):
    image_path = item["image_path"]
    paragraph = item["narration"]

    try:
        image = load_img(image_path)
    except Exception as e:
        print(f"[Error] Cannot read image {image_path}: {e}")
        return None

    result = {
        "image": image_path,
        "text": paragraph,
        "index": idx
    }

    for eval_type, prompt_template in [
        ("character_consistency", prompt_character),
        ("time_location_consistency", prompt_time_location),
        ("non_character_entity_consistency", prompt_non_character)
    ]:
        print(f"[{idx}] calculating {eval_type}...")

        transcript = [{"role": "system", "content": ""}, {"role": "user", "content": []}]
        transcript[-1]["content"].append(load_text(prompt_template))
        transcript[-1]["content"].append(load_text(paragraph))
        transcript[-1]["content"].append(load_img(image_path))

        temp_start = 0.0
        max_retry = 5
        score = None
        analysis = ""

        while True:
            try:
                response = gptv_query(transcript=transcript, temp=temp_start)
                print(f"[{idx}] [{eval_type}] Raw Response:\n{response}")
                for line in response.strip().split("\n"):
                    if line.lower().startswith("analysis:"):
                        analysis = line.split(":", 1)[1].strip()
                matches = re.findall(r"(score|Score):\s*[a-zA-Z]*\s*(\d+)", response)
                if len(matches) == 0:
                    raise ValueError("No valid score found")
                score = int(matches[0][1])
                break
            except Exception as e:
                temp_start += 0.1
                max_retry -= 1
                if max_retry == 0:
                    score = 0
                    analysis = "[Fallback] Could not extract analysis or score."
                    break

        result[eval_type] = {
            "score": score,
            "analysis": analysis
        }

    return result



for story in sorted(os.listdir(json_dir)):
    if story.endswith(".json") and 'subtitles' in story:
        json_path = os.path.join(json_dir, story)
        from pathlib import Path
        path = Path(json_path)
        stem = path.stem  # 不含扩展名
        id = stem.split('_')[-2]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        print(f"processing {story}")

        max_workers = 20  # 设置你想要的并发请求数

        print(f"Starting parallel evaluation with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_item, idx, item, prompt_character, prompt_time_location,
                                prompt_non_character): idx
                for idx, item in enumerate(data, start=1)
            }

            for future in tqdm(as_completed(futures), total=len(data)):
                result = future.result()
                if result:
                    results.append(result)


        out_path1 = f"{output_dir}/analysis_story{id}.json"
        # 保存为 JSON 文件
        with open(out_path1, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"analisis have been saved to {out_path1}")
        # === 6. 计算平均分并输出 ===
        score_summary = {}
        score_totals = {
            "character_consistency": 0,
            "time_location_consistency": 0,
            "non_character_entity_consistency": 0
        }
        count = 0

        for res in results:
            idx = str(res["index"])
            char_score = res["character_consistency"]["score"]
            time_score = res["time_location_consistency"]["score"]
            nonchar_score = res["non_character_entity_consistency"]["score"]

            # # 跳过无效分数
            # if None in [char_score, time_score, nonchar_score]:
            #     continue

            count += 1
            score_summary[idx] = {
                "character_consistency": char_score,
                "time_location_consistency": time_score,
                "non_character_entity_consistency": nonchar_score
            }

            score_totals["character_consistency"] += char_score
            score_totals["time_location_consistency"] += time_score
            score_totals["non_character_entity_consistency"] += nonchar_score

        avg_score_dict = {
            key: round(score_totals[key] / count, 2) if count > 0 else 0
            for key in score_totals
        }

        score_summary["avg_score_dict"] = avg_score_dict

        # 保存输出
        out_path = f"{output_dir}/score_story{id}.json"
        # 保存为 JSON 文件
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(score_summary, f, ensure_ascii=False, indent=2)



