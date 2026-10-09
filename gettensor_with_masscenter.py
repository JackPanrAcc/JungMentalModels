# Author: jackpanr 261006
# Descript: 
#   请先阅读并运行gettensor_load_intrinsicdim.py 提取大模型Intrinsic Dimensionality 相关的数据后。
#   获得Intrinsic Dimensionality 后，本脚本文件中可以计算：
#     1. 思考复杂度 = FreeenergyMaxuperank(embeddings, maxuperank)
#     2. 自省肤浅度 = FreeenergyIntrospect(embeddings, introspect)
#     3. 其中符号自由能公式 Freeenergy(base,this) = [1 - cosSimilarity(tensorBase-centerOfMassBase, tensorThis-centerOfMassThis)] * sqrt(tensorDim)
#
# 本脚本文件的处理逻辑
#   第一段部分：从《重心文件》中读出三个重心，从filename_of_cases 读 word 计算“思考复杂度”和“自省肤浅度”然后输出完整报告
#   第二段部分：从《张量缓存库文件》中读高频词的三个张量，基于《高频词表》中登记的权重计算三个语义空间的重心并保存
#   第三段部分：在 WORD_FIRST>=0 时，轮询高频词表并提取对应的 embeddings、 maxuperank、 introspect 三层张量并保存在《张量缓存库文件》
#
# 高频词表来源是 https://blcu.edu.cn。下载后保存在本地 multi_domain_total_word_freq.txt。文件中每行一个词，包含单词和频率两个属性、用半角逗号分隔
#
#
# NOTE1: find_sublist_indices(){full_text.find()} 因为查单次位置。所以，提示词中多次出现 word 查询是错误的。严肃的做法是查字典
#        本脚本中，token 左侧只有["你","说","的","是"]等 word，遂逐一确认token positions 之后硬编码
# NOTE2: 高频词表第167行[一样]。分词器划分的token是[你说/的/是一/样/。/对/...']。无法确定词向量
# NOTE3: 高频词表第179行[一起]。分词器划分的token是[你说/的/是一/起/。/对/...']。无法确定词向量
#


import torch
import torch.nn.functional as FUNC

from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json

import os
import sys
import gc




#这是第三部分中读取高频词表的起始行，手工修改值将决定第三部分是否会运行
WORD_FIRST = 0

# 固定的输入提示语模板
default_text = "你说的是{PAD4REPLACE}。对某些人来说这是错误的，或者说，有些时候有些有些场景这句话大概率是错误的，那么你想的是什么？"

# 使用gettensor_load_intrinsicdim.py 脚本提取的大模型数据
HIDDEN_DIM = 3584
model_name = "Qwen/Qwen2.5-7B-Instruct"
intrinsicdim_layer_index_embeddings = 0   # layer_number+1 即hiddenstate 数组中的index
intrinsicdim_layer_index_maxuperank = 7   # layer_number+1 即hiddenstate 数组中的index
intrinsicdim_layer_index_introspect = 23  # layer_number+1 即hiddenstate 数组中的index
intrinsicdim_maxuperank = 3471.44

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1" 


# 文件名配置项
pyscript_dir           = os.path.dirname(os.path.abspath(__file__))
path_output            = os.path.join(pyscript_dir, "multi_domain_total_word_freq.report.txt")
path_cases             = os.path.join(pyscript_dir, "multi_domain_total_word_freq.cases.json")
path_center_embeddings = os.path.join(pyscript_dir, "multi_domain_total_word_freq.masscenter.embeddings.txt")
path_center_maxuperank = os.path.join(pyscript_dir, "multi_domain_total_word_freq.masscenter.maxuperank.txt")
path_center_introspect = os.path.join(pyscript_dir, "multi_domain_total_word_freq.masscenter.introspect.txt")
path_tensor            = os.path.join(pyscript_dir, "multi_domain_total_word_freq.tensor.txt")
path_freq              = os.path.join(pyscript_dir, "multi_domain_total_word_freq.txt")

cached_tensors_embeddings_map = {}
cached_tensors_maxuperank_map = {}
cached_tensors_introspect_map = {}
if os.path.exists(path_tensor):
    print(f"正在加载三层张量缓存库: {os.path.basename(path_tensor)} ...")
    with open(path_tensor, "r", encoding="utf-8") as f:
        for line in f:
            # 【核心重构】：优先按 \t 切分
            parts_tabs = line.strip().split("\t")
            if len(parts_tabs) == 4:
                w = str(parts_tabs[0]).strip()
                
                # 模块内部依然按空格切分出 3584 维向量
                emb_floats = parts_tabs[1].split()
                max_floats = parts_tabs[2].split()
                int_floats = parts_tabs[3].split()
                
                if len(emb_floats) == HIDDEN_DIM and len(max_floats) == HIDDEN_DIM and len(int_floats) == HIDDEN_DIM:
                    cached_tensors_embeddings_map[w] = torch.tensor([float(x) for x in emb_floats], dtype=torch.float16)
                    cached_tensors_maxuperank_map[w] = torch.tensor([float(x) for x in max_floats], dtype=torch.float16)
                    cached_tensors_introspect_map[w] = torch.tensor([float(x) for x in int_floats], dtype=torch.float16)
    print(f"张量缓存库中一共加载了 {len(cached_tensors_embeddings_map)} 条记录")



# 辅助函数：检索分词器引起的多Token 组合
def find_sublist_indices(tokenizer, input_ids_tensor, target_word):
    if hasattr(input_ids_tensor, "tolist"):
        readable_ids = input_ids_tensor.cpu().tolist()
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

    char_start = full_text.find(target)
    if char_start == -1:
        # 降级尝试：将分词生成的常见占位符（如 Qwen 的 Ġ 或特殊空格）做平滑处理后再找一次
        full_text_clean = "".join([t.replace(" ", " ").replace("Ġ", "") for t in readable_tokens])
        char_start_clean = full_text_clean.find(target)
        if char_start_clean == -1:
            return []

        char_end_clean = char_start_clean + len(target)
        # 重新平滑映射边界
        res = []
        current_char_idx = 0
        for token_idx, token in enumerate(readable_tokens):
            t_clean = token.replace(" ", " ").replace("Ġ", "")
            t_len = len(t_clean)
            t_start = current_char_idx
            t_end = current_char_idx + t_len
            current_char_idx = t_end
            if max(t_start, char_start_clean) < min(t_end, char_end_clean):
                res.append(token_idx)
        return res

    char_end = char_start + len(target)
    res = []
    for token_idx, (t_start, t_end) in enumerate(token_boundaries):
        if max(t_start, char_start) < min(t_end, char_end):
            res.append(token_idx)
    return res



# 辅助函数：加载高频词
def load_frequency_table(path, limit=500):
    words_freq_map = []
    if not os.path.exists(path): return words_freq_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("token") or cleaned.startswith("word"): continue

            parts = cleaned.split(',') # 和《高频词表》格式保持一致
            if len(parts) < 2: continue
            try:
                word_item = str(parts[0]).strip()
                freq_val = float(parts[1].strip()) 
                words_freq_map.append({"word": word_item, "frequency": freq_val})
                if limit and len(words_freq_map) >= limit: break
            except (ValueError, IndexError):
                continue
    return words_freq_map



# ==================== 第一部分：加载测试用例，求解符号自由能 ====================
if not all([os.path.exists(path_center_embeddings), os.path.exists(path_center_maxuperank), os.path.exists(path_center_introspect)]):
    print("\n【第一部分】 未执行。原因是未能读取《语义空间的重心》文件")
elif not os.path.exists(path_cases):
    print("\n【第一部分】 未执行。原因是无法读取《测试用例配置文件 gettensor_test_cases.json》")
else:
    with open(path_center_embeddings, "r", encoding="utf-8") as f_emb, \
         open(path_center_maxuperank, "r", encoding="utf-8") as f_max, \
         open(path_center_introspect, "r", encoding="utf-8") as f_int:
        c_emb = torch.tensor([float(x) for x in f_emb.read().strip().split()], dtype=torch.float16)
        c_max = torch.tensor([float(x) for x in f_max.read().strip().split()], dtype=torch.float16)
        c_int = torch.tensor([float(x) for x in f_int.read().strip().split()], dtype=torch.float16)

    with open(path_cases, "r", encoding="utf-8") as f_case:
        test_cases = json.load(f_case)
    print("\n【第一部分】 " + "="*70 + "\n读取用例文件，开逐行计算“思考复杂度”和“自省肤浅度”...")

    # 初始化大模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True) #intel版mac 只能用cpu。请按自己硬件优化
    model.eval()
    sqrt_tensor_dim = math.sqrt(model.config.hidden_size)
    target_device = model.device

    final_freeenergy_report = {}
    for case in test_cases:
        traceid = str(case.get("traceid", case.get("debug_case_traceid", "")))
        word = case.get("word", case.get("debug_case_word", ""))
        if not word: continue

        status = "model_inference"
        if word in cached_tensors_embeddings_map and word in cached_tensors_maxuperank_map and word in cached_tensors_introspect_map:
            t_emb = cached_tensors_embeddings_map[word]
            t_max = cached_tensors_maxuperank_map[word]
            t_int = cached_tensors_introspect_map[word]
            status = "cached_hit"
        else:
            # 缓存未命中，激活大模型前向传播
            inputs = tokenizer(default_text.replace("{PAD4REPLACE}", word), return_tensors="pt")
            # 确保输入 Tensor 移至模型对应设备（如果是GPU加速则必须）
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            positions = find_sublist_indices(tokenizer, inputs["input_ids"], word)
            if not positions: continue

            # --- 如果出现multiple tokens，则打印 work、 positions、 tokens ---
            if len(positions)>1 or word in ["你说的是","你说的","说的是","你说","说的","的是","你","说","的","是"]:
                input_ids_list = inputs["input_ids"][0].cpu().tolist() 
                all_tokens_diagnostic = []
                for multipletokenidx, multipletokentid in enumerate(input_ids_list):
                    try:
                        token_str = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(multipletokentid)])
                    except Exception:
                        token_str = tokenizer.decode([multipletokentid], errors="replace")
                    token_str = token_str.replace("\n", "\\n").replace("\r", "\\r")
                    all_tokens_diagnostic.append(f"[{multipletokenidx}:{multipletokentid}]='{token_str}'")

                print(f"\n" + "="*50)
                print(f"[诊断 - multiple token]")
                print(f"  ├─ Word : '{word}'")
                print(f"  ├─ Positions: {positions}")
                print(f"  └─ Tokens:")
                for i in range(0, len(all_tokens_diagnostic), 4):
                    print("      " + "  |  ".join(all_tokens_diagnostic[i:i+4]))
                print("="*50 + "\n")
            # 组合共有10种"你说的是","你说的","说的是","你说*","说的*","的是","你*","说*","的*","是*"。BCC高频词表 中有带星号的6种 
            if word in ["你说"] and default_text.startswith("你说的是{PAD4REPLACE}"): positions=[0]
            if word in ["说的"] and default_text.startswith("你说的是{PAD4REPLACE}"): positions=[2]
            if word in ["你","说","的","是"] and default_text.startswith("你说的是{PAD4REPLACE}"): positions=[2]
            # ----------------------------


            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # 提取对应层特征并计算均值
            t_emb = outputs.hidden_states[intrinsicdim_layer_index_embeddings][0, positions, :].mean(dim=0)
            t_max = outputs.hidden_states[intrinsicdim_layer_index_maxuperank][0, positions, :].mean(dim=0)
            t_int = outputs.hidden_states[intrinsicdim_layer_index_introspect][0, positions, :].mean(dim=0)

        # ====================================================================
        # 安全防御升级：强行将所有外来张量同步至 target_device，并统一转化为 float32
        #             注意有可能张量缓存库文件中是一个精度，切平台实时跑出来是另精度
        # ====================================================================
        t_emb_f32 = t_emb.to(device=target_device, dtype=torch.float32)
        t_max_f32 = t_max.to(device=target_device, dtype=torch.float32)
        t_int_f32 = t_int.to(device=target_device, dtype=torch.float32)

        c_emb_f32 = c_emb.to(device=target_device, dtype=torch.float32)
        c_max_f32 = c_max.to(device=target_device, dtype=torch.float32)
        c_int_f32 = c_int.to(device=target_device, dtype=torch.float32)

        s_emb = t_emb_f32 - c_emb_f32
        s_max = t_max_f32 - c_max_f32
        s_int = t_int_f32 - c_int_f32

        cos_sim_max = FUNC.cosine_similarity(s_emb.unsqueeze(0), s_max.unsqueeze(0), dim=1).item()
        cos_sim_int = FUNC.cosine_similarity(s_emb.unsqueeze(0), s_int.unsqueeze(0), dim=1).item()

        fe_maxuperank = (1.0 - cos_sim_max) * sqrt_tensor_dim
        fe_introspect = (1.0 - cos_sim_int) * sqrt_tensor_dim

        final_freeenergy_report[traceid] = {
            "word": word, "status": status,
            "cos_sim_max": cos_sim_max, "fe_maxuperank": fe_maxuperank,
            "cos_sim_int": cos_sim_int, "fe_introspect": fe_introspect
        }
        print(f"  【{word}】 思考复杂度: {fe_maxuperank:.4f} | 自省肤浅度: {fe_introspect:.4f} ({status})")

    if final_freeenergy_report:
        with open(path_output, "w", encoding="utf-8") as f_out:
            f_out.write(f"=== 符号自由能综合分析报告 ===\n")
            f_out.write(f"{'TraceID':<10}{'Word':<12}{'Type':<14}{'CosSimMax':<12}{'Complexity':<14}{'CosSimInt':<12}{'Introspect':<14}\n")
            f_out.write("-" * 92 + "\n")
            for tid, val in final_freeenergy_report.items():
                f_out.write(f"{tid:<10}{val['word']:<12}{val['status']:<14}{val['cos_sim_max']:<12.6f}{val['fe_maxuperank']:<14.4f}{val['cos_sim_int']:<12.6f}{val['fe_introspect']:<14.4f}\n")
        print(f"\n【第一部分】 处理成功，报告已保存: {path_output}")

    print("【第一部分】符号自由能分析完毕，终止脚本运行。")
    sys.exit(0) # 既然三个层的重心均已算完，那么没有必要运行第二、三部分代码




# ==================== 第二部分：从本地张量缓存库中合成三组语义重心 ====================
top500_freq_data = load_frequency_table(path_freq, limit=500)
if top500_freq_data:
    missing_words = [item["word"] for item in top500_freq_data if item["word"] not in cached_tensors_introspect_map]
    if len(missing_words) > 0:
        print(f"\n【第二部分】 未执行。原因是张量缓存库内尚缺 {len(missing_words)} 个高频词的张量")
        print(f" 缺失的 word 示例，请在《高频词表》中查看行号并修改 WORD_FIRST: {missing_words[:10]}")
    else:
        print(f"\n【第二部分】 开始计算语义重心...")

        # 读取数据通过 .float() 统一转换为安全、高精度的float32 进行计算
        v_emb = torch.stack([cached_tensors_embeddings_map[item["word"]] for item in top500_freq_data]).float()
        v_max = torch.stack([cached_tensors_maxuperank_map[item["word"]] for item in top500_freq_data]).float()
        v_int = torch.stack([cached_tensors_introspect_map[item["word"]] for item in top500_freq_data]).float()

        cache_device = v_emb.device
        w_list = [item["frequency"] for item in top500_freq_data]
        sum_w = sum(w_list)
        
        w_tensor = torch.tensor([w / sum_w for w in w_list], dtype=torch.float32, device=cache_device)
        com_emb = torch.mv(v_emb.t(), w_tensor)
        com_max = torch.mv(v_max.t(), w_tensor)
        com_int = torch.mv(v_int.t(), w_tensor)

        # 结果高精度落盘
        with open(path_center_embeddings, "w", encoding="utf-8") as f_e, \
             open(path_center_maxuperank, "w", encoding="utf-8") as f_m, \
             open(path_center_introspect, "w", encoding="utf-8") as f_i:
            f_e.write(" ".join([f"{x.item():.8f}" for x in com_emb]))
            f_m.write(" ".join([f"{x.item():.8f}" for x in com_max]))
            f_i.write(" ".join([f"{x.item():.8f}" for x in com_int]))

        print("【第二部分】重心更新完毕，终止脚本运行。")
        sys.exit(0) # 既然张量缓存库已完整，那么没有必要运行第三部分代码



# ==================== 第三部分：从大模型轮询三层张量并保存在张量缓存库文件 ====================
if WORD_FIRST >= 0:
    print("\n" + "="*70 + "\n【第三部分】激活大模型前向传播，多特征空间层并发安全提取与追加入库...")
    all_words_to_run = load_frequency_table(path_freq, limit=None)
    if all_words_to_run:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True) #intel版mac 只能用cpu。请按自己硬件优化
        model.eval()
        target_device = model.device

        already_extracted_words = set(cached_tensors_introspect_map.keys())
        for idx, item in enumerate(all_words_to_run[WORD_FIRST:]):
            word = item["word"]
            global_idx = WORD_FIRST + idx
            if word in already_extracted_words: continue

            inputs = tokenizer(default_text.replace("{PAD4REPLACE}", word), return_tensors="pt")
            # 确保输入 Tensor 移至模型对应设备（如果是GPU加速则必须）
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            positions = find_sublist_indices(tokenizer, inputs["input_ids"], word)
            if not positions: continue

            # --- 如果出现multiple tokens，则打印 work、 positions、 tokens ---
            if len(positions)>1 or word in ["你说的是","你说的","说的是","你说","说的","的是","你","说","的","是"]:
                input_ids_list = inputs["input_ids"][0].cpu().tolist() 
                all_tokens_diagnostic = []
                for multipletokenidx, multipletokentid in enumerate(input_ids_list):
                    try:
                        token_str = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(multipletokentid)])
                    except Exception:
                        token_str = tokenizer.decode([multipletokentid], errors="replace")
                    token_str = token_str.replace("\n", "\\n").replace("\r", "\\r")
                    all_tokens_diagnostic.append(f"[{multipletokenidx}:{multipletokentid}]='{token_str}'")

                print(f"\n" + "="*50)
                print(f"[诊断 - multiple token]")
                print(f"  ├─ Word : '{word}'")
                print(f"  ├─ Positions: {positions}")
                print(f"  └─ Tokens:")
                for i in range(0, len(all_tokens_diagnostic), 4):
                    print("      " + "  |  ".join(all_tokens_diagnostic[i:i+4]))
                print("="*50 + "\n")
            # 组合共有10种"你说的是","你说的","说的是","你说*","说的*","的是","你*","说*","的*","是*"。BCC高频词表 中有带星号的6种 
            if word in ["你说"] and default_text.startswith("你说的是{PAD4REPLACE}"): positions=[0]
            if word in ["说的"] and default_text.startswith("你说的是{PAD4REPLACE}"): positions=[2]
            if word in ["你","说","的","是"] and default_text.startswith("你说的是{PAD4REPLACE}"): positions=[2]
            # ----------------------------


            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            tensor_emb = outputs.hidden_states[intrinsicdim_layer_index_embeddings][0, positions, :].mean(dim=0).cpu()
            tensor_max = outputs.hidden_states[intrinsicdim_layer_index_maxuperank][0, positions, :].mean(dim=0).cpu()
            tensor_int = outputs.hidden_states[intrinsicdim_layer_index_introspect][0, positions, :].mean(dim=0).cpu()

            s_floats_emb = " ".join([f"{x.item():.8f}" for x in tensor_emb])
            s_floats_max = " ".join([f"{x.item():.8f}" for x in tensor_max])
            s_floats_int = " ".join([f"{x.item():.8f}" for x in tensor_int])

            record_line = f"{word}\t{s_floats_emb}\t{s_floats_max}\t{s_floats_int}\n"
            parts_check = record_line.strip().split("\t")
            if len(parts_check) == 4:
                if len(parts_check[1].split()) == HIDDEN_DIM and \
                   len(parts_check[2].split()) == HIDDEN_DIM and \
                   len(parts_check[3].split()) == HIDDEN_DIM:
                    with open(path_tensor, "a", encoding="utf-8") as f_tensor:
                        f_tensor.write(record_line)
                        f_tensor.flush()

            already_extracted_words.add(word)
            print(f" -> 第{global_idx}行 【{word}】 成功抓取了三张量并完成入库到本地张量缓存库")
            gc.collect()

            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()



