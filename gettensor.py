# Author: jackpanr 261006
# Descript: 保持提示语一致，直接对比目标单词在深层的隐层张量特征（完美兼容多Token切词）

import torch
import torch.nn.functional as FUNC

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import hashlib
import json

import sys
import os


# 配置数据范例，且可以从 json 配置文件中加载 debug_case_list
default_text = "你说的是{PAD4REPLACE}。对某些人来说这是错误的，或者说，有些时候有些有些场景这句话大概率是错误的，那么你想的是什么？"
debug_case_list = [
  {"debug_case_traceid": "1", "debug_case_word": "社会"},
  {"debug_case_traceid": "2", "debug_case_word": "分子"},
  {"debug_case_traceid": "3", "debug_case_word": "确定"},
]

# 配置大模型离线/国内镜像环境加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"  # 本地已有缓存

filename_of_config = "gettensor_debug_case_list.json"
filename_of_output = "gettensor.txt"
try:
  pyscript_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(pyscript_dir, filename_of_config), "r", encoding="utf-8") as f:
    debug_case_list = json.load(f)
except FileNotFoundError:
    debug_case_list = debug_case_list


# 辅助函数：从数组生成 md5 码以备校验
def build_md5_for_array(arr, use_float32: bool = False) -> str:
  if isinstance(arr, torch.Tensor):
    processed_arr = arr.detach().cpu().numpy()
  elif isinstance(arr, np.ndarray):
    processed_arr = arr
  else:
    raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(arr)}")
  if processed_arr.size == 0:
    return hashlib.md5(b"").hexdigest()

  target_dtype = np.float32 if use_float32 else np.float64
  processed_arr = processed_arr.astype(target_dtype).ravel()
    
  if processed_arr.dtype.byteorder == '=':
  if sys.byteorder == 'little':
    processed_arr = processed_arr.byteswap()
  elif processed_arr.dtype.byteorder == '<':
    processed_arr = processed_arr.byteswap()

  buf = processed_arr.tobytes(order='C')
  h = hashlib.md5()
  h.update(buf)
  return h.hexdigest()


# 辅助函数：在Token 列表中寻找目标Token 所有的物理索引位置
def find_sublist_indices(tokenizer, input_ids_tensor, target_word):
    # 确保input_ids 为标准的一维Python 列表
    if hasattr(input_ids_tensor, "tolist"):
      readable_ids = input_ids_tensor.tolist()
    else:
      readable_ids = input_ids_tensor
    if len(readable_ids) > 0 and isinstance(readable_ids[0], list):
      readable_ids = readable_ids[0]

    # 逐个 Token 解码
    readable_tokens = [tokenizer.decode([tid]) for tid in readable_ids]
    target = target_word.strip()

    # === 基于全句还原的物理索引反查 ===
    full_text = ""
    token_boundaries = [] 
    for token in readable_tokens:
      start_idx = len(full_text)
      full_text += token
      end_idx = len(full_text)
      token_boundaries.append((start_idx, end_idx))

    char_start = full_text.find(target)
    if char_start == -1:
      return [] 
    char_end = char_start + len(target)
    res = []
    for token_idx, (t_start, t_end) in enumerate(token_boundaries):
      if max(t_start, char_start) < min(t_end, char_end):
        res.append(token_idx)
    return res



# ========== 以下代码先加载大模型后循环执行每一行 debug case ==========
model_name = "Qwen/Qwen2.5-7B-Instruct"
level_pos = 25  # 提取深度语义层

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)

# 用来缓存每个 case 提取出的目标词深层向量，以便后续做交叉对比
extracted_vectors_cache = {}
for debug_case in debug_case_list:
  debug_case_traceid = debug_case.get("debug_case_traceid", "")
  debug_case_word = debug_case.get("debug_case_word", "")
  debug_case_note = debug_case.get("debug_case_note", "")
  proc_skip = debug_case.get("procSkip", 0)  
  text_wanna = debug_case.get("text", "")
  if len(text_wanna) < 1:
    text_wanna = default_text
  if proc_skip != 0:
    continue  # 忽略本行json

  # 1. 拼装并转化为Token IDs
  text_wanna = text_wanna.replace("{PAD4REPLACE}", debug_case_word)
  inputs = tokenizer(text_wanna, return_tensors="pt")
  input_ids_list = inputs["input_ids"][0].tolist()

  # 2. 对目标单词进行编码，获取其子 Token IDs
  word_encoded = tokenizer.encode(debug_case_word, add_special_tokens=False)

  # 3. 动态检索目标词在整句话中的 Token 物理索引位置（支持多 Token 紧密聚合）
  token_positions = find_sublist_indices(tokenizer, inputs["input_ids"], debug_case_word)
  if not token_positions:
    print(f"\n##### 报错了：Case{debug_case_traceid} 在句中未匹配到单词「{debug_case_word}」的 Token 结构。")
    continue

  # 4. 前向传播提取隐层状态
  with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
  all_hidden = outputs.hidden_states
  
  # 5. 提取静态词向量层（Layer 0）和深层语义层（Layer 25）
  layer_embedding = all_hidden[0][0]  # 形状: [Seq_len, Hidden_dim]
  layer_wanna = all_hidden[level_pos][0]  # 形状: [Seq_len, Hidden_dim]
  
  # 6. 对多 Token 物理位置的向量进行均值融合
  tensor_embedding = layer_embedding[token_positions, :].mean(dim=0)
  tensor_wanna = layer_wanna[token_positions, :].mean(dim=0)
  extracted_vectors_cache[debug_case_traceid] = {
      "word": debug_case_word,
      "tensor": tensor_wanna
  }

  # 7. 计算当前词的【层间演变相似度】（Layer 0 vs Layer 25）并记录 MD5
  cos_sim_self = FUNC.cosine_similarity(tensor_embedding.unsqueeze(0), tensor_wanna.unsqueeze(0))
  checkpoint = build_md5_for_array(tensor_embedding)
  log_str = (
    f'\n{{case:{debug_case_traceid}, word:"{debug_case_word}", '
    f'debug_case_note:"{debug_case_note}", '
    f'embedding_checkpoint:"{checkpoint}", cos_similarity:{cos_sim_self.item():.16f},}}'
  )
  print(log_str)
  with open(os.path.join(pyscript_dir, filename_of_output), "a", encoding="utf-8") as flog:
    flog.write(log_str)


# ========== 以下代码打印总报表 ==========
print("\n" + "="*60 + "\n【交叉对比分析】同一句型下，不同目标词在第 25 层的几何空间相似度：")
cache_keys = list(extracted_vectors_cache.keys())
for i in range(len(cache_keys)):
  for j in range(i + 1, len(cache_keys)):
    case_id1, case_id2 = cache_keys[i], cache_keys[j]
    data1 = extracted_vectors_cache[case_id1]
    data2 = extracted_vectors_cache[case_id2]

    # 计算两个不同词在完全相同的上下文中演变出的深层向量相似度
    cos_sim_cross = FUNC.cosine_similarity(data1["tensor"].unsqueeze(0), data2["tensor"].unsqueeze(0)).item()
    cross_log = f' -> Case {case_id1}({data1["word"]}) vs Case {case_id2}({data2["word"]}) 相似度: {cos_sim_cross:.16f}'
    print(cross_log)
    with open(os.path.join(pyscript_dir, filename_of_output), "a", encoding="utf-8") as flog:
     flog.write("\n" + cross_log)


