import argparse
import time
from requests.api import patch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html
import string

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


 
translation_table = str.maketrans(punctuation_dict)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()

    model.to(device='cuda',  dtype=torch.bfloat16)

    # TODO vary old codes, NEED del 
    image_processor = BlipImageEvalProcessor(image_size=1024)

    image_processor_high =  BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256

    image = load_image(args.image_file)

    w, h = image.size
    # print(image.size)
    
    if args.type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    if args.box:
        bbox = eval(args.box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
            bbox[2] = int(bbox[2]/w*1000)
            bbox[3] = int(bbox[3]/h*1000)
        if args.type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '

    if args.color:
        if args.type == 'format':
            qs = '[' + args.color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + args.color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)

    inputs = tokenizer([prompt])

    # vary old codes, no use
    image_1 = image.copy()
    image_tensor = image_processor(image)

    image_tensor_1 = image_processor_high(image_1)

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.confidences = []

        start = time.time()
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams = 1,
            no_repeat_ngram_size = 20,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )
        end = time.time()

        # print('output_ids:', output_ids)
        print(f"\n 耗时 {end - start}s")

        # 获取生成的所有新token
        generated_tokens = output_ids[0, input_ids.shape[1]:].tolist()

        confidences = []
        token_result = []
        # 逐个解码每个token并关联置信度
        for idx, (token, confidence) in enumerate(zip(generated_tokens, model.confidences[1:])):
            token_str = tokenizer.decode(token, skip_special_tokens=True)
            print(f"Token {idx+1} ({token_str}) —— Confidence: {confidence.item():.4f}")
            confidences.append(f"{confidence.item():.4f}")  # 收集置信度
            token_result.append(token_str)

        if args.render:
            print('==============rendering===============')

            generated_tokens = output_ids[0, input_ids.shape[1]:].tolist()
            # print("Generated Tokens:", generated_tokens)

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            print('outputs:', outputs)

            # 将结果保存到txt文件
            with open("results.txt", "a", encoding="utf-8") as f:
                # 使用制表符分隔格式：图片路径 | 生成文本 | 置信度列表
                f.write(f"{args.image_file}\n推理结果：{outputs}\nToken：{','.join(token_result)}\n置信度：{','.join(confidences)}\n耗时：{end - start:.4f}s\n\n")

            if '**kern' in outputs:
                import verovio
                from cairosvg import svg2png
                import cv2
                import numpy as np
                tk = verovio.toolkit()
                tk.loadData(outputs)
                tk.setOptions({"pageWidth": 2100, "footer": 'none',
               'barLineWidth': 0.5, 'beamMaxSlope': 15,
               'staffLineWidth': 0.2, 'spacingStaff': 6})
                tk.getPageCount()
                svg = tk.renderToSVG()
                svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")

                svg_to_html(svg, "./results/demo.html")

            if args.type == 'format' and '**kern' not in outputs:

                
                if  '\\begin{tikzpicture}' not in outputs:
                    html_path = "./render_tools/" + "/content-mmd-to-html.html"
                    html_path_2 = "./results/demo.html"
                    right_num = outputs.count('\\right')
                    left_num = outputs.count('\left')

                    if right_num != left_num:
                        outputs = outputs.replace('\left(', '(').replace('\\right)', ')').replace('\left[', '[').replace('\\right]', ']').replace('\left{', '{').replace('\\right}', '}').replace('\left|', '|').replace('\\right|', '|').replace('\left.', '.').replace('\\right.', '.')


                    outputs = outputs.replace('"', '``').replace('$', '')

                    outputs_list = outputs.split('\n')
                    gt= ''
                    for out in outputs_list:
                        gt +=  '"' + out.replace('\\', '\\\\') + r'\n' + '"' + '+' + '\n' 
                    
                    gt = gt[:-2]

                    with open(html_path, 'r') as web_f:
                        lines = web_f.read()
                        lines = lines.split("const text =")
                        new_web = lines[0] + 'const text ='  + gt  + lines[1]
                else:
                    html_path = "./render_tools/" + "/tikz.html"
                    html_path_2 = "./results/demo.html"
                    outputs = outputs.translate(translation_table)
                    outputs_list = outputs.split('\n')
                    gt= ''
                    for out in outputs_list:
                        if out:
                            if '\\begin{tikzpicture}' not in out and '\\end{tikzpicture}' not in out:
                                while out[-1] == ' ':
                                    out = out[:-1]
                                    if out is None:
                                        break
    
                                if out:
                                    if out[-1] != ';':
                                        gt += out[:-1] + ';\n'
                                    else:
                                        gt += out + '\n'
                            else:
                                gt += out + '\n'


                    with open(html_path, 'r') as web_f:
                        lines = web_f.read()
                        lines = lines.split("const text =")
                        new_web = lines[0] + gt + lines[1]

                with open(html_path_2, 'w') as web_f_new:
                    web_f_new.write(new_web)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-name", type=str, default="/home/zhoukefan/GOT-OCR2.0/GOT-OCR-2.0-master/GOT_Weights")
#     parser.add_argument("--image-file", type=str, default='/home/zhoukefan/GOT-OCR2.0/imgs/nums.jpg', required=True)
#     parser.add_argument("--type", type=str, default='ocr', required=True)
#     parser.add_argument("--box", type=str, default= '')
#     parser.add_argument("--color", type=str, default= '')
#     parser.add_argument("--render", action='store_true')
#     args = parser.parse_args()

#     eval_model(args)

if __name__ == "__main__":
    class Args:
        def __init__(self, model_name, image_file) -> None:
            self.model_name = model_name    # "/home/zhoukefan/GOT-OCR2.0/ocr_flip_test/checkpoint-16000"
            self.image_file = image_file     # '/home/zhoukefan/GOT-OCR2.0/imgs/2.png'  # 修改为你实际的图片路径
            self.type = 'ocr'
            self.box = ''
            self.color = ''
            self.render = True  # 根据需求设置 True/False
    
    from pathlib import Path

    def process_cut_folder(model_name, images_path):
        supported_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

        # 遍历cut文件夹
        for filename in os.listdir(images_path):
            filepath = os.path.join(images_path, filename)
            
            # 过滤非图片文件
            if not filename.lower().endswith(supported_exts):
                continue

            # 创建参数对象
            args = Args(
                model_name=model_name,
                image_file=filepath
            )

            try:
                # 执行处理
                eval_model(args)
                print(f"成功处理: {filename}")
            except Exception as e:
                print(f"处理 {filename} 失败: {str(e)}")

    process_cut_folder(
        "/home/zhoukefan/GOT-OCR2.0/ocr_flip_test/checkpoint-16000", 
        '/home/zhoukefan/GOT-OCR2.0/imgs/16/cut'
    )
    # args = Args("/home/zhoukefan/GOT-OCR2.0/ocr_flip_test/checkpoint-16000", '/home/zhoukefan/GOT-OCR2.0/imgs/IMG_20250220_141756_180.jpg')
    # eval_model(args)
