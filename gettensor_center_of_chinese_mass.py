# Author: jackpanr 261006
# Descript: 
#   Freeenergy = [1-cosSimilarity(embedding0 - centerOfMass00percent, tensor80percent - centerOfMass80percent)] * sqrt(tensorDim)
#   本脚本用于首先下载高频词表，然后用大模型计算《 0%深度上中文语义空间的重心 centerOfMass00percent》
#。                                       《80%深度上中文语义空间的重心 centerOfMass80percent》
#   高频词表是这个版本 https://blcu.edu.cn

import torch
import torch.nn.functional as FUNC
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import os
import math

WORD_FIRST = 0  # 【可手工修改】第二部分读取词频表的起始行数索引（0代表从头开始，方便断点续传）

# 语义心智深度控制（Layer 0 为初始层，Layer 25 为 80% 深度语义层）
level_pos_0 = 0
level_pos_80 = 25  
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 固定的输入提示语模板（保持上下文绝对一致）
default_text = "你说的是{PAD4REPLACE}。对某些人来说这是错误的，或者说，有些时候有些有些场景这句话大概率是错误的，那么你想的是什么？"

# 持久化落盘文件名严格对齐（彻底修复重名覆盖Bug）
filename_of_freq = "multi_domain_total_word_freq.txt"
filename_of_tensor = "multi_domain_total_word_freq.tensor.txt"
filename_of_center00 = "multi_domain_total_word_freq.output00.txt"
filename_of_center80 = "multi_domain_total_word_freq.output80.txt"
filename_of_cases = "gettensor_test_cases.json"
filename_of_output = "gettensor_freeenergy_report.txt"

pyscript_dir = os.path.dirname(os.path.abspath(__file__))
path_freq = os.path.join(pyscript_dir, filename_of_freq)
path_tensor = os.path.join(pyscript_dir, filename_of_tensor)
path_center00 = os.path.join(pyscript_dir, filename_of_center00)
path_center80 = os.path.join(pyscript_dir, filename_of_center80)
path_cases = os.path.join(pyscript_dir, filename_of_cases)
path_output = os.path.join(pyscript_dir, filename_of_output)

# 环境加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1" 


# 辅助函数：精准的文本反向物理窗口滑窗查找函数（彻底防御二维Batch列表嵌套Bug）
def find_sublist_indices(tokenizer, input_ids_tensor, target_word):
    if hasattr(input_ids_tensor, "tolist"):
        readable_ids = input_ids_tensor.tolist()
    else:
        readable_ids = input_ids_tensor

    if len(readable_ids) > 0 and isinstance(readable_ids[0], list):
        readable_ids = readable_ids[0]

    readable_tokens = [tokenizer.decode([tid]) for tid in readable_ids]
    target = target_word.strip()

    full_text = ""
    token_boundaries = [] 
    for token in readable_tokens:
        start_idx = len(full_text)
        full_text += token
        end_idx = len(full_text)
        token_boundaries.append((start_idx, end_idx))

    char_start = full_text.rfind(target)
    if char_start == -1:
        return [] 
    char_end = char_start + len(target)
    res = []
    for token_idx, (t_start, t_end) in enumerate(token_boundaries):
        if max(t_start, char_start) < min(t_end, char_end):
            res.append(token_idx)
    return res


# 辅助函数：安全加载词频表
def load_frequency_table(path, limit=500):
    words_freq_map = []
    if not os.path.exists(path):
        return words_freq_map
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("token,count") or cleaned.startswith("word,"): 
            continue  
        
        parts = cleaned.replace(",", " ").split()
        if len(parts) >= 2:
            word = parts[0].strip()
            try:
                freq = float(parts[1])
                words_freq_map.append({"word": word, "frequency": freq})
                count += 1
                if limit and count >= limit:
                    break
            except ValueError:
                continue
    return words_freq_map


# ==================== 🪐 第一部分：从本地缓存中合成双层中文语义空间重心 ====================
print("\n" + "="*70 + "\n【第一阶段】正在扫描本地缓存以验证 Top 500 重心数据完整性...")

# 1. 读取前 500 个高频词的词频 (将 limit 校准为 500，以匹配引力重心大数定律需求)
top500_freq_data = load_frequency_table(path_freq, limit=500)
if not top500_freq_data:
    raise FileNotFoundError(f"错误：未在 {filename_of_freq} 中读取到有效的词频数据！请检查文件是否存在。")
    
# 2. 从本地已经跑出来的双层张量库反查数据
cached_tensors_0_map = {}
cached_tensors_80_map = {}

if os.path.exists(path_tensor):
    with open(path_tensor, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # 刚性判定：一行必须由 1个单词 + 3584维层0 + 3584维层25 组成
            if len(parts) == 1 + 3584 * 2:
                w = parts[0]
                vec_0 = torch.tensor([float(x) for x in parts[1:3585]], dtype=torch.float16)
                vec_80 = torch.tensor([float(x) for x in parts[3585:]], dtype=torch.float16)
                cached_tensors_0_map[w] = vec_0
                cached_tensors_80_map[w] = vec_80

# 检查高频词是否全员到齐
missing_words = [item["word"] for item in top500_freq_data if item["word"] not in cached_tensors_80_map]

is_ready_for_center = False
if len(missing_words) > 0:
    print(f"提示：当前本地张量库中尚未集齐 top500 全部高频词（目前还差 {len(missing_words)} 个）。")
    print(f" -> 缺失示例: {missing_words[:6]}。系统将自动激活大模型断点续传机制补齐数据...")
else:
    is_ready_for_center = True
    print(f"✅ top500 核心词数据全员到齐！开始基于引力质量加权合成双层绝对重心...")
    
    vectors_0_list = [cached_tensors_0_map[item["word"]] for item in top500_freq_data]
    vectors_80_list = [cached_tensors_80_map[item["word"]] for item in top500_freq_data]
    weights_list = [item["frequency"] for item in top500_freq_data]
    
    total_freq = sum(weights_list)
    normalized_weights = [w / total_freq for w in weights_list]
    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float16)

    # 🪐 矩阵引力质量相乘：(所有高频词向量 * 真实词频占比系数) 累加
    centerOfMass00percent = torch.mv(torch.stack(vectors_0_list).t(), weights_tensor)
    centerOfMass80percent = torch.mv(torch.stack(vectors_80_list).t(), weights_tensor)
    
    # 5. 【修复落盘】同步写入保存独立的 00 和 80 两个文件
    with open(path_center00, "w", encoding="utf-8") as f00, open(path_center80, "w", encoding="utf-8") as f80:
        f00.write(" ".join([f"{x.item():.8f}" for x in centerOfMass00percent]))
        f80.write(" ".join([f"{x.item():.8f}" for x in centerOfMass80percent]))
        
    print(f"\n========================================================")
    print(f"🎉 终期大捷！双层中文语义空间绝对重心张量已成功生成：")
    print(f" -> 0% 初始层重心已保存至: {filename_of_center00} (均值: {centerOfMass00percent.mean().item():.6f})")
    print(f" -> 80% 深度层重心已保存至: {filename_of_center80} (均值: {centerOfMass80percent.mean().item():.6f})")
    print(f"========================================================")


# ==================== 🚀 第二部分：大模型表征批量提取/断点续传 ====================
if not is_ready_for_center:
    print("\n" + "="*70 + "\n【第二阶段启动】激活大模型批量提取/断点续传流水线...")

    already_extracted_words = set(cached_tensors_80_map.keys())
    all_words_to_run = load_frequency_table(path_freq, limit=None) 
    if not all_words_to_run:
        raise FileNotFoundError(f"错误：未在当前目录下找到词频表 {filename_of_freq} ！")

    print("正在加载 Tokenizer 与大模型权重...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    current_dim = model.config.hidden_size

    print(f"\n🚀 开始从第 {WORD_FIRST} 行起执行模型前向传播计算...")
    sliced_words_list = all_words_to_run[WORD_FIRST:]

    for idx, item in enumerate(sliced_words_list):
        word = item["word"]
        global_idx = WORD_FIRST + idx
        
        if word in already_extracted_words:
            continue
            
        text_wanna = default_text.replace("{PAD4REPLACE}", word)
        inputs = tokenizer(text_wanna, return_tensors="pt")
        positions = find_sublist_indices(tokenizer, inputs["input_ids"], word)
        
        if not positions:
            print(f" ⚠️ 行 {global_idx}: 未能匹配到 「{word}」 的 Token 结构，跳过。")
            continue
            
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # 【双层闭环核心改动】：同步安全提取第 0 层和第 25 层的池化向量
        tensor_0 = outputs.hidden_states[level_pos_0][0, positions, :].mean(dim=0)
        tensor_25 = outputs.hidden_states[level_pos_80][0, positions, :].mean(dim=0)
        
        # 3. 将两层向量合并为一长行序列化文本
        str_floats_0 = " ".join([f"{x.item():.8f}" for x in tensor_0])
        str_floats_25 = " ".join([f"{x.item():.8f}" for x in tensor_25])
        record_line = f"{word} {str_floats_0} {str_floats_25}\n"
        
        with open(path_tensor, "a", encoding="utf-8") as f_tensor:
            f_tensor.write(record_line)
            
        already_extracted_words.add(word)
        print(f" -> [行 {global_idx}] 成功捕获并合并追加落盘: 【{word}】")

    print(f"\n✨ 批量提取数据完成！请重新运行本脚本一秒生成双层重心文件。")


# ==================== 🏆 第三部分：加载测试案例，求解去噪后的绝对语义自由能 ====================
if is_ready_for_center:
    print("\n" + "="*70 + "\n【第三阶段启动】读取单义测试词，执行双平移流形消噪并求解 Freeenergy...")
    
    if not os.path.exists(path_cases):
        # 如果文件不存在，内存动态注入高解耦测试种子，防止代码闪退
        test_cases = [{"traceid": "1", "word": "寸"}, {"traceid": "2", "word": "社会"}, {"traceid": "3", "word": "方法"}]
    else:
        with open(path_cases, "r", encoding="utf-8") as f_case:
            test_cases = json.load(f_case)
            
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    current_dim = model.config.hidden_size
    
    final_freeenergy_report = {}
    for case in test_cases:
        traceid = str(case.get("traceid", case.get("debug_case_traceid", "")))
        word = case.get("word", case.get("debug_case_word", ""))
        if not word: continue
        
        text = default_text.replace("{PAD4REPLACE}", word)
        inputs = tokenizer(text, return_tensors="pt")
        positions = find_sublist_indices(tokenizer, inputs["input_ids"], word)
        if not positions: continue
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        embedding0 = outputs.hidden_states[level_pos_0][0, positions, :].mean(dim=0)
        tensor80percent = outputs.hidden_states[level_pos_80][0, positions, :].mean(dim=0)
