# Author: jackpanr 261007
# Descript: 
#   Freeenergy = [1-cosSimilarity(embedding - centerOfMass0percent, tensor80percent - centerOfMass80percent)] * sqrt(tensorDim)
#   本脚本用于首先下载高频词表，然后用大模型计算《 0%深度上中文语义空间的重心 centerOfMass0percent》《80%深度上中文语义空间的重心 centerOfMass80percent》
#
#   高频词表是这个版本 https://blcu.edu.cn。下载后保存在本地multi_domain_total_word_freq.txt
#
# 第一段部分：从filename_of_cases 读word 计算Freeenergy
# 第二段部分：从multi_domain_total_word_freq.tensor.txt 文件中读0percent 和80percent 语义空间的高频词张量，基于multi_domain_total_word_freq.txt 文件中的词频权重算重心并保存
# 第三段部分：在WORD_FIRST 大于等于0时，依序从multi_domain_total_word_freq.txt 文件中WORD_FIRST 行开始读出高频词，从大模型查询对应的张量并保存在multi_domain_total_word_freq.tensor.txt
#

import torch
import torch.nn.functional as FUNC

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
import json

import os



#这是第三部分中读取高频词表的起始行，手工修改值将决定第三部分是否会运行
WORD_FIRST = 16

# 固定的输入提示语模板
default_text = "你说的是{PAD4REPLACE}。对某些人来说这是错误的，或者说，有些时候有些有些场景这句话大概率是错误的，那么你想的是什么？"

# 语义心智深度控制（Layer 0 为初始层，Layer 25 为 80% 抽象度语义空间）
level_pos_0 = 0
level_pos_80 = 25
level_pos_100 = 28
model_name = "Qwen/Qwen2.5-7B-Instruct"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1" 


# 文件名配置项
filename_of_output = "gettensor_with_masscenter.report.txt" # 从cases 读到word 列表之后计算自由能并保存在此文件
filename_of_cases = "gettensor_with_masscenter.cases.json" # 依据本文件第3行《符号自由能公式》逐行计算word 的自由能
filename_of_center00 = "multi_domain_total_word_freq.masscenter.00percent.txt" # embedding 中文语义空间的重心
filename_of_center80 = "multi_domain_total_word_freq.masscenter.80percent.txt" # 80percent 中文语义空间的重心
filename_of_tensor = "multi_domain_total_word_freq.tensor.txt" # embedding、80percent 高频词张量缓存库 
filename_of_freq = "multi_domain_total_word_freq.txt" # 从权威机构抓取的“高频词”和“词频”

pyscript_dir = os.path.dirname(os.path.abspath(__file__))
path_output = os.path.join(pyscript_dir, filename_of_output)
path_cases = os.path.join(pyscript_dir, filename_of_cases)
path_center00 = os.path.join(pyscript_dir, filename_of_center00)
path_center80 = os.path.join(pyscript_dir, filename_of_center80)
path_tensor = os.path.join(pyscript_dir, filename_of_tensor)
path_freq = os.path.join(pyscript_dir, filename_of_freq)

cached_tensors_0_map = {}
cached_tensors_80_map = {}
if os.path.exists(path_tensor):
    print(f"正在加载张量缓存库: {filename_of_tensor} ...")
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
    print(f"从张量缓存库中共加载了 {len(cached_tensors_80_map)} 条记录")


# 辅助函数：查找拆词器所引起的token 组合
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


# 辅助函数：加载高频词
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



# ==================== 第一部分：加载测试用例，求解符号自由能之后进行保存 ====================
if not os.path.exists(path_center00) or not os.path.exists(path_center80):
    print("\n【第一部分】 未执行。原因是未能找到《语义空间的重心》文件")
elif not os.path.exists(path_cases):
    print("\n【第一部分】 未执行。原因是无法读取《测试用例配置文件gettensor_test_cases.json》")
else:
    with open(path_cases, "r", encoding="utf-8") as f_case:
        test_cases = json.load(f_case)

    print("\n【第一部分】 " + "="*70 + "\n读case 配置文件，对每个 word 计算 Freeenergy...")
    # 从重心文件中读取语义空间的重心坐标
    with open(path_center00, "r", encoding="utf-8") as f00, open(path_center80, "r", encoding="utf-8") as f80:
        centerOfMass0percent = torch.tensor([float(x) for x in f00.read().strip().split()], dtype=torch.float16)
        centerOfMass80percent = torch.tensor([float(x) for x in f80.read().strip().split()], dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True) #intel版mac 只能用cpu。请按自己硬件优化
    tensorDim = model.config.hidden_size 
    sqrt_tensor_dim = math.sqrt(tensorDim)

    final_freeenergy_report = {}
    for case in test_cases:
        traceid = str(case.get("traceid", case.get("debug_case_traceid", "")))
        word = case.get("word", case.get("debug_case_word", ""))
        if not word: continue

        embedding0 = None
        tensor80percent = None
        status = "model_inference"
        if word in cached_tensors_0_map and word in cached_tensors_80_map:
            # 成功命中缓存！直接使用张量缓存库中缓存的张量
            embedding0 = cached_tensors_0_map[word]
            tensor80percent = cached_tensors_80_map[word]
            status = "cached_hit"
        else:
            # 缓存未命中，通过大模型前向传播网络实时提取
            text = default_text.replace("{PAD4REPLACE}", word)
            inputs = tokenizer(text, return_tensors="pt")
            positions = find_sublist_indices(tokenizer, inputs["input_ids"], word)
            if not positions: 
                print(f"[TraceID {traceid}]: 测试词 【{word}】 未能匹配到 Token 结构，跳过。")
                continue
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            embedding0 = outputs.hidden_states[level_pos_0][0, positions, :].mean(dim=0)
            tensor80percent = outputs.hidden_states[level_pos_80][0, positions, :].mean(dim=0)

        # 1. 减去对应层级的语义空间重心
        shifted_0 = embedding0.to(torch.float32) - centerOfMass0percent.to(torch.float32)
        shifted_80 = tensor80percent.to(torch.float32) - centerOfMass80percent.to(torch.float32)

        # 2. 计算消噪后向量的余弦相似度
        cos_sim = FUNC.cosine_similarity(shifted_0.unsqueeze(0), shifted_80.unsqueeze(0), dim=1).item()

        # 3. 自由能公式：Freeenergy = (1 - cosSim) * sqrt(tensorDim)
        freeenergy = (1.0 - cos_sim) * sqrt_tensor_dim

        # 4. 登记到最终报告字典
        final_freeenergy_report[traceid] = {
            "word": word,
            "status": status,
            "cosine_similarity": cos_sim,
            "freeenergy": freeenergy
        }
        print(f"    [结果] 【{word}】 消噪余弦相似度: {cos_sim:.6f}, 语义自由能freeenergy: {freeenergy:.4f}")

    # === 将所有计算报告保存到文件备查 ===
    if final_freeenergy_report:
        with open(path_output, "w", encoding="utf-8") as f_out:
            f_out.write(f"=== 符号自由能报告 (model: {model_name})(tensorDim: {tensorDim})(level: {level_pos_80}/{level_pos_100}) ===\n")
            f_out.write(f"{'TraceID':<10}{'Word':<12}{'Type':<18}{'CosSim':<12}{'FreeEnergy':<12}\n")
            f_out.write("-" * 65 + "\n")
            for tid, val in final_freeenergy_report.items():
                f_out.write(f"{tid:<10}{val['word']:<12}{val['status']:<18}{val['cosine_similarity']:<12.6f}{val['freeenergy']:<12.4f}\n")
        print(f"\n【第一部分】 第一部分执行完毕，自由能计算结果已写入: {filename_of_output}")
    raise FileNotFoundError(f"【第一部分】执行完毕。脚本退出执行状态")




# ==================== 第二部分：从本地缓存中合成双层中文语义空间重心 ====================
# 1. 读取前 top500 个高频词的词频
top500_freq_data = load_frequency_table(path_freq, limit=500)
if not top500_freq_data:
    raise FileNotFoundError(f"【第二部分】未执行。原因是无法读取《词频表 {filename_of_freq} 》")

# 2. 检查高频词是否完整，完整则加权后保存相关思考深度的语义空间重心
missing_words = [item["word"] for item in top500_freq_data if item["word"] not in cached_tensors_80_map]
if len(missing_words) > 0:
    print(f"\n【第二部分】 未执行。原因是本地张量库中尚未缓存 top500高频词（目前还差{len(missing_words)} 个）")
    print(f" 缺失示例: {missing_words[:6]}。系统将自动激活大模型断点续传机制补齐数据...")
else:
    print(f"\n【第二部分】 top500高频词 张量已经缓存！现在开始计算80%思考深度的语义空间重心...")
    vectors_0_list = [cached_tensors_0_map[item["word"]] for item in top500_freq_data]
    vectors_80_list = [cached_tensors_80_map[item["word"]] for item in top500_freq_data]
    weights_list = [item["frequency"] for item in top500_freq_data]

    # 3. 矩阵引力质量相乘：(所有高频词向量 * 真实词频占比系数) 累加
    total_freq = sum(weights_list)
    normalized_weights = [w / total_freq for w in weights_list]
    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float16)
    centerOfMass00percent = torch.mv(torch.stack(vectors_0_list).t(), weights_tensor)
    centerOfMass80percent = torch.mv(torch.stack(vectors_80_list).t(), weights_tensor)

    # 4. 写入 0percent 和 80percent 文件
    with open(path_center00, "w", encoding="utf-8") as f00, open(path_center80, "w", encoding="utf-8") as f80:
        f00.write(" ".join([f"{x.item():.8f}" for x in centerOfMass00percent]))
        f80.write(" ".join([f"{x.item():.8f}" for x in centerOfMass80percent]))
    print(f"\n========================================================")
    print(f"【第二部分】0% 语义空间的重心位置保存在: {filename_of_center00} (位置: {centerOfMass00percent.mean().item():.6f})")
    print(f"【第二部分】80% 语义空间的重心位置保存在: {filename_of_center80} (位置: {centerOfMass80percent.mean().item():.6f})")
    print(f"========================================================")




# ==================== 第三部分：从大模型轮询张量并append 到multi_domain_total_word_freq.tensor.txt ====================
if WORD_FIRST>=0 :
    print("\n" + "="*70 + "\n【第三部分】从大模型轮询张量并append 到日志文件...")

    already_extracted_words = set(cached_tensors_80_map.keys())
    all_words_to_run = load_frequency_table(path_freq, limit=None) 
    if not all_words_to_run:
        raise FileNotFoundError(f"【第三部分】未执行。原因是无法读取《词频表 {filename_of_freq} 》")

    print("【第三部分】正在加载 Tokenizer 与大模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True) #intel版mac 只能用cpu。请按自己硬件优化
    current_dim = model.config.hidden_size

    print(f"\n【第三部分】开始从第 {WORD_FIRST} 行开始执行大模型的前向传播计算...")
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
            print(f"行 {global_idx}: 未能匹配到 「{word}」 的 Token 结构，跳过。")
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 【双层闭环核心改动】：同步安全提取第 0 层和第 25 层的池化向量
        tensor_0 = outputs.hidden_states[level_pos_0][0, positions, :].mean(dim=0)
        tensor_25 = outputs.hidden_states[level_pos_80][0, positions, :].mean(dim=0)

        # 将两层向量合并为一长行序列化文本
        str_floats_0 = " ".join([f"{x.item():.8f}" for x in tensor_0])
        str_floats_25 = " ".join([f"{x.item():.8f}" for x in tensor_25])
        record_line = f"{word} {str_floats_0} {str_floats_25}\n"
        with open(path_tensor, "a", encoding="utf-8") as f_tensor:
            f_tensor.write(record_line)

        already_extracted_words.add(word)
        print(f" -> [行 {global_idx}] 成功捕获并合并追加: 【{word}】")
    print(f"\n【第三部分】 批量提取数据完成！请删除重心文件之后重新运行本脚本刷新重心文件")




