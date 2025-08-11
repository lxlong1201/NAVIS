# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform

from cli.SparkTTS import SparkTTS

model_dir = "/data1/lxl_data/sparkTTS"
prompt_text = "它是一只可爱的小兔子，眼睛大大的，耳朵长长的，但是它有一个小小的烦恼，它非常胆小，特别是在课堂上"
prompt_speech_path = "sample_speech/sample 2.wav"
# save_dir = '/home/user/lxl/story-adapter'


"""Perform TTS inference and save the generated audio."""

# Convert device argument to torch.device
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    # macOS with MPS support (Apple Silicon)
    device = torch.device(f"mps:{0}")
    logging.info(f"Using MPS device: {device}")
elif torch.cuda.is_available():
    # System with CUDA support
    device = torch.device(f"cuda:{0}")
    logging.info(f"Using CUDA device: {device}")
else:
    # Fall back to CPU
    device = torch.device("cpu")
    logging.info("GPU acceleration not available, using CPU")

# Initialize the model
model = SparkTTS(model_dir, device)

# Generate unique filename using timestamp
# timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# save_path = os.path.join(save_dir, f"{timestamp}.wav")
#
# logging.info("Starting inference...")

# # Perform inference and save the output audio
# with torch.no_grad():
#     wav = model.inference(
#         text,
#         prompt_speech_path,
#         prompt_text=prompt_text
#         # gender=args.gender,
#         # pitch=args.pitch,
#         # speed=args.speed,
#     )
#     sf.write(save_path, wav, samplerate=16000)
#
# logging.info(f"Audio saved at: {save_path}")



from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, CompositeAudioClip
import torch
import os
import json
import re
from TTS.api import TTS

os.environ["TTS_CACHE_PATH"] = "/data1/lxl_data/story-ad/tts"

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio


# download and load all models
# preload_models()
# Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(TTS().list_models())
# # Init TTS 并移动到目标 device
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def split_sentences(text):
    # 保留分隔符并分割
    parts = re.split(r'([。\.])', text)
    sentences = []
    for idx in range(0, len(parts) - 1, 2):
        sent = parts[idx].strip() + parts[idx + 1].strip()
        if sent:
            sentences.append(sent)
    # 如果最后一段没有标点，单独添加
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences

stories = [1, 2, 3, 4]
for id, story in enumerate(stories):
    json_path = f"/home/user/lxl/story-adapter/subtitles25/story{id+1}.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        text = item['narration']
        scene_number = item['scene_number']
        text = text.replace('"', '')
        subtitles = split_sentences(text)
        item["subtitles"] = subtitles
        subtitle_wav = []
        # 为该 subtitles 下的每个句子生成 wav
        for j, subtitle in enumerate(subtitles):
            ############bark##############################
            wav_dir = f"/home/user/lxl/story-adapter/story_test/story_{id+1}/story_wav"
            if not os.path.exists(wav_dir):
                os.mkdir(wav_dir)
            out_path = os.path.join(wav_dir, f"output_{scene_number}_{j}.wav")
            # save audio to disk
            # audio_array = generate_audio(subtitle, history_prompt="v2/en_speaker_9")
            # write_wav(out_path, SAMPLE_RATE, audio_array)
            ############bark###############################
            subtitle_wav.append(out_path)
            # tts.tts_to_file(
            #     text=subtitle,
            #     speaker_wav="/home/user/lxl/story-adapter/sample.wav",
            #     language="en",
            #     file_path=out_path
            # )
            print(f"Input text: {subtitle}")
            with torch.no_grad():
                wav = model.inference(
                    subtitle,
                    prompt_speech_path,
                    prompt_text=prompt_text
                    # gender=args.gender,
                    # pitch=args.pitch,
                    # speed=args.speed,
                )
                sf.write(out_path, wav, samplerate=16000)
            print(f"Generated: {out_path}")
        item['subtitle_wav'] = subtitle_wav

    # 加入元素后在写入json
    json_final = f'/home/user/lxl/story-adapter/story_test/story_{id+1}/story.json'
    with open(json_final, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    ###########################moviepy工作流#############################

    output_path = f"/home/user/lxl/story-adapter/story_test/story_{id+1}/test.mp4"
    # Zoom 配置
    max_zoom = 1.2
    fps = 24

    # 加载 JSON 文件
    with open(json_final, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_clips = []

    for item in data:
        img_path = item["image_path"]
        wavs = item["subtitle_wav"]
        subtitles = item["subtitles"]
        assert len(wavs) == len(subtitles), f"{base} 的音频与字幕数量不一致"

        img_clip = ImageClip(img_path)
        w, h = img_clip.size

        # 加载音频并计算总时长
        audio_clips = []
        durations = []
        total_duration = 0
        for wav_path in wavs:
            audio_clip = AudioFileClip(wav_path)
            duration = audio_clip.duration
            audio_clips.append(audio_clip)
            durations.append(duration)
            total_duration += duration

        # 创建持续 zoom-in 动画
        zoomed = (
            img_clip
            .with_duration(total_duration)
            .resized(lambda t: 1 + (max_zoom - 1) * (t / total_duration))
            .with_position("center")
        )

        # 添加字幕与音频
        subtitle_clips = []
        audio_clips_with_start = []
        composite_audio = None
        start = 0
        for subtitle_text, audio_clip, duration in zip(subtitles, audio_clips, durations):
            subtitle = (
                TextClip(
                    text=subtitle_text,
                    font_size=40,
                    color='white',
                    method='caption',
                    text_align="center",
                    size=(w, None)
                )
                .with_duration(duration)
                .with_start(start)
                .with_position(('center', 'bottom'))
            )
            subtitle_clips.append(subtitle)

            audio_clip = audio_clip.with_start(start)
            audio_clips_with_start.append(audio_clip)

            # composite_audio = audio_clip if composite_audio is None else composite_audio.with_end(start).with_start(0).audio_fadeout(0) + audio_clip
            start += duration

        composite_audio = CompositeAudioClip(audio_clips_with_start)
        # 合成视频
        composite = CompositeVideoClip([zoomed] + subtitle_clips, size=(w, h), bg_color=(0, 0, 0)).with_audio(composite_audio)
        final_clips.append(composite)

    # 合并所有片段
    final_video = concatenate_videoclips(final_clips, method='compose')

    # 输出视频
    final_video.write_videofile(
        output_path,
        fps=fps,
        audio=True,
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )












