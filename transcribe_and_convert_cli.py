import subprocess
import os
import sys
import time
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from convert_output import convert
import json

import json

import json

def convert_to_json(data, write_to_file=False, file_path='output.json'):
    """
    将类似JSON格式的原始数据转换为标准JSON字符串,并可选择将结果写入文件
    
    :param data: 包含原始数据的列表
    :param write_to_file: 是否将结果写入文件,默认为False
    :param file_path: 输出文件路径,默认为'output.json'
    :return: 格式化后的JSON字符串
    """
    # 创建一个新列表来存储格式化后的数据
    formatted_data = []
    
    # 遍历原始数据
    for item in data:
        # 将元组转换为列表,并处理可能存在的None值
        timestamp = [value for value in item['timestamp'] if value is not None]
        
        # 创建一个新的字典,使用双引号括起属性名称
        formatted_item = {
            "timestamp": timestamp,
            "text": item['text']
        }
        
        # 将格式化后的字典添加到新列表中
        formatted_data.append(formatted_item)
    
    # 将格式化后的数据转换为JSON字符串
    json_str = json.dumps(formatted_data, ensure_ascii=False)
    
    # 如果需要,将JSON字符串写入文件
    if write_to_file:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    return json_str

def transcribe_and_convert(audio_dir):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/large-v2",
        # openai/whisper-base  openai/whisper-large-v3
        torch_dtype=torch.float32,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith((".mp3", ".wav", ".m4a")):
                file_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                json_output_path = os.path.join(root, base_name + '_raw'+ ".json")

                print(f"Transcribing {file_path}...")
                # Record the start time
                start_time = time.time()

                outputs = pipe(
                    file_path,
                    chunk_length_s=30,
                    batch_size=24,
                    return_timestamps=True,
                )

                # Print the recognized subtitle text
                for output in outputs:
                    # Calculate the elapsed time during the transcription
                    elapsed_time = time.time() - start_time
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")

                # Calculate the processing time
                end_time = time.time()
                processing_time = end_time - start_time
                print(f"Processing time: {processing_time:.2f} seconds")

                # 指定要输出的 JSON 文件路径
                output_file_path = os.path.join(root, base_name + ".json")

                # 打印输出
                # print("Output data:", outputs)
                # 将字典内容写入 JSON 文件
                with open(output_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(outputs, json_file, ensure_ascii=False, indent=4)

                print(f"Converting {json_output_path} to SRT...")
                output_format = 'srt'
                output_dir = root

                convert(output_file_path, output_format, output_dir, verbose=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_and_convert.py <audio_dir>")
        sys.exit(1)

    audio_dir = sys.argv[1]
    transcribe_and_convert(audio_dir)
