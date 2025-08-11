# NAVIS
 A modular, multimodal pipeline NAVIS for generating and aligning narration with visual stories. 
## Quick Start 

### Installation
The project is built with Python 3.10.14, PyTorch 2.2.2. CUDA 12.1, cuDNN 8.9.02
For installing, follow these instructions:
~~~
# create new anaconda env1
conda create -n StoryAdapter python=3.10
conda activate StoryAdapter 

# install packages
pip install -r requirements.txt
pip install TTS
pip install moviepy==2.2.1
pip install numpy==1.23.2
pip install IPython
~~~
You also need to create env2 according to [Qwen](https://github.com/QwenLM/Qwen2.5-VL) and prepare the weight of Qwen2.5-VL-32B
### Download the checkpoint
- downloading [RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0/tree/main) put it into "ckpt/story_ad/RealVisXL_V4.0"
- downloading [clip_image_encoder](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models/image_encoder) put it into "ckpt/IP-Adapter/sdxl_models/image_encoder"
- downloading [ip-adapter_sdxl](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin?download=true) put it into "ckpt/ip-adapter_sdxl.bin"

### Running Demo

~~~
conda activate StoryAdapter
python story_vis.py

conda activate env2
python story_gen.py

conda activate StoryAdapter
python video_gen.py
~~~
## Evaluation
For our image-narration score, you can run as below
~~~
conda activate StoryAdapter
python eva_align.py
python eva_story.py
~~~
For the image-promt score, you can according to [Vistorybench](https://github.com/ViStoryBench/vistorybench)
- downloading [weight](https://drive.google.com/file/d/1SETgjkj6oUIbjgwxgtXw2I2t4quRzG-3/view?usp=drive_link) for CSD evaluation
