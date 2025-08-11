from sympy.physics.units import temperature
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
import torch
import json
import os

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "ckpt/qwenvl7",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# default processor
processor = AutoProcessor.from_pretrained("ckpt/qwenvl7")
bench_dir = 'subtitles_story'

# 1. 全局定义 System Prompt
SYSTEM_PROMPT = """
You will be given the previous story keyframes and their corresponding narrations, and a new scene image with its description.
Each image is paired **one-to-one** with the narration that follows it — they describe the same moment.

Task Definition:
You are a visual storyteller. You are continuing a story based on prior narration-image pairs and a new scene pairs.
Your task is to write exactly one paragraph (10–20 words), Use the new scene pairs as the main basis, while referring to prior narrations for consistency.


PRINCIPLES:
P1. Complete Story Structure: events must flow toward an emotional resolution.
P2. Natural Temporal and Causal Flow: each paragraph grows out of the last; no hints of future scenes.
P3. Emotional Depth and Character Growth: show feelings/thoughts that drive the next action.
P4. Thematic Unity & Character Consistency: stick to one theme; characters behave consistently.
P5. Gentle & Simple Language: age‑appropriate (5–8 years), soft wording.

Rules:
- Write one paragraph with 10–20 **English words**.
- Do **not** introduce any characters **not** present in the scene description.
- Do **not** repeat or rephrase any full sentence from previous narrations.

- Build a coherent story with a beginning, middle, climax, and ending, and the events must progress toward an emotional resolution.
- **Use action and dialogue** to move the story forward, Include **at least 1 line of dialogue or thought**.
- Maintain character identity, emotion, and intent continuity.
- If the characters was introduced by **full name once** in the previous story, then refer to them using **he/she/they**.
- Never describe the previous scenes or merge multiple scenes.
- Avoid visual details; focus on action and dialogue.

- Each scene must clearly grow out of the one before—there should be **no sudden jumps** or unexplained changes.
- Use linking emotions and motivations to transition. Avoid hard scene cuts. Let us see the *thought* that leads to the *next action*.
- Use words to show motivation and consequences, to make the story feel smooth and connected.
- **Show a clear cause→effect link** with the immediately preceding scene:
  1. **Reference** the previous narration-image pairs.
  2. **Explain** how that outcome/emotion **triggers** the new action or thought.

- **Ensure character consistency:** no actions or attitudes should contradict what’s been established.
- **Focus primarily on the current prompt and image:** center narration on the new scene and its description.

EXAMPLE 1:

Narration 1:  
Lila peeked out the train window, eyes wide. "It’s really happening," she whispered, gripping her ticket.

Narration 2:  
When the train jerked to a stop, she tumbled into her seat. "Oops!" she laughed, brushing off her coat.

Narration 3:  
Grandma waved from the platform. Lila ran straight into her arms. "You still smell like peppermint," she giggled.

Narration 4:  
Later, in the backyard, chickens flapped around her feet. "Wait, I don’t have snacks!" she yelped, backing away.

Narration 5:  
Still catching her breath, Lila tiptoed toward the barn. "Just a cow," she mumbled, "How scary can it be?"

Description for frame:  
Lila drops the metal bucket as the cow snorts loudly. She steps back, then frowns and picks it up.

Generated continuation (10–20 words):  
"Whoa!" she cried, stumbling back. But then she chuckled, "Alright, alright—I get it," and bent to grab the bucket.

EXAMPLE 2:

Narration 1
The little squirrel pressed his nose to the window. Outside looked cold, but he was listening. “Is spring here?” he whispered.

Narration 2
The big squirrel came over with a scarf and hat. “Let’s go find out,” he said with a smile.

Narration 3
They stepped outside. The trees were bare, and frost sparkled on the ground.

Narration 4
The little squirrel cupped his ears. “Just wind,” he sighed. “Still no spring.”

Narration 5
They walked through the forest. Leaves crunched under their feet. Everything was quiet.

Description for frame:  
The little squirrel stops and points at a small green sprout near a rock.

Generated continuation (10–20 words):  
Suddenly, the little squirrel stopped. “Look!” he pointed. A tiny green sprout peeked out near a rock.

"""


def build_qwen_messages(images, prompts, narrations, frame_idx, length):
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    if frame_idx == 0:
        # ---------- 首帧 ----------
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Description for the first frame: {prompts[0]}"},
                {"type": "text",
                 "text": "Now write narration for frame 1 (10–20 words), following the rules and examples above."}
            ]
        })

    else:
        # ---------- 非首帧 ----------
        content = []
        start = max(0, frame_idx - length)
        for i in range(start, frame_idx):
            content.append({"type": f"image {i + 1}", "image": images[i]})
            content.append({"type": "text", "text": f"Narration {i + 1}: {narrations[i]}"})

        content.append({"type": f"Current image {frame_idx}", "image": images[frame_idx]})
        content.append({"type": "text", "text": f"Description for frame {frame_idx + 1}: {prompts[frame_idx]}"})
        content.append({
            "type": "text",
            "text": f"Now write narration for frame {frame_idx + 1}/{len(prompts)} (10–20 words), following rules and examples above."
        })

        messages.append({"role": "user", "content": content})

    return messages


context_length = [7]
for length in context_length:
    for story in sorted(os.listdir(bench_dir)):
        if not story.endswith('.json') or 'subtitles' in story:
            continue

        json_path = os.path.join(bench_dir, story)
        path = Path(json_path)
        idx = path.stem.split('_')[-2]
        seed = path.stem.split('_')[-1]

        # 读取原始 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提前收集所有图像路径和 prompts
        images = [item['image_path'] for item in data]
        prompts = [item['prompt'] for item in data]

        # 用于存储前一帧生成的 narration
        generated_narrations = []
        re_story_txt = ""


        for frame_idx, item in enumerate(data):
            messages = build_qwen_messages(
                images, prompts, generated_narrations, frame_idx, length
            )

            # 文本 + 视觉预处理
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # print(f"frame_idx_{frame_idx}: {text}")
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # 模型推理生成
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                num_beams=1,
                temperature=0.7,
                top_p=0.9
            )
            # 去除输入前缀
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )


            item['narration'] = output_text[0]
            generated_narrations.append(output_text[0])
            re_story_txt += output_text[0]
            print(f"[Frame {frame_idx + 1}] {output_text[0]}")


        story_txt_path = f"{bench_dir}/subtitles_{length}_{idx}_{seed}.txt"
        with open(story_txt_path, 'w', encoding='utf-8') as f:
            f.write(re_story_txt)

        json_final = f"{bench_dir}/subtitles_{length}_{idx}_{seed}.json"
        with open(json_final, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Finished refining story {idx}: outputs saved.")
