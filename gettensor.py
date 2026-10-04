# Author: jackpanr 261004
# Descript: get cosine_similarity between the tensor and embedding.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as FUNC
import torch
import json
import os

default_token_wanna = "自由"
default_text = "符号自由能，语义空间"
debug_case_list = [
  {"debug_case_traceid": "1", "debug_case_word": "语义"},
  {"debug_case_traceid": "2", "debug_case_word": "语义2"},
]
pyscript_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(pyscript_dir,"gettensor_debug_case_list.json"), "r", encoding="utf-8") as f:
    debug_case_list = json.load(f)


# ========== 以下代码先加载大模型后循环执行每一行debug case ==========
model_name = "Qwen/Qwen2.5-7B-Instruct"
level_pos = 25  # Qwen2.5的Transformer 是28层。这里提取的是24号层

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)  # 我笔记本是intel 版mac。只能跑CPU版Transformer
for debug_case in debug_case_list:
  debug_case_traceid = debug_case.get("debug_case_traceid", "")
  debug_case_word = debug_case.get("debug_case_word", "")
  debug_case_note = debug_case.get("debug_case_note", "")
  token_wanna = debug_case.get("token_wanna", "")
  if len(token_wanna)<1 :
    token_wanna = default_token_wanna
  text_wanna = debug_case.get("text", "")
  if len(text_wanna)<1 :
    text_wanna = default_text

  text_wanna = text_wanna.replace("{PAD4REPLACE}", debug_case_word)
  inputs = tokenizer(text_wanna, return_tensors="pt")
  readable_ids = inputs["input_ids"][0]
  readable_tokens = [tokenizer.decode([tid]) for tid in readable_ids]
  try:
    token_pos = readable_tokens.index(token_wanna)
  except ValueError:
    print("\n##### ##### ###### ###### ###### 报错了：case", debug_case_traceid, "没找到token。word=", debug_case_word, ", token_wanna=", token_wanna, "")
    print("token列表：", readable_tokens)
    token_pos = -1
    continue  # 找不到token则跳过case。注意：此处输出的不是json 格式！

  with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
  all_hidden = outputs.hidden_states
  layer_embedding = all_hidden[0]
  tensor_embedding = layer_embedding[0, token_pos, :]
  layer_wanna = all_hidden[level_pos]
  tensor_wanna = layer_wanna[0, token_pos, :]
  # print(f"\n[{debug_case_id}] 第{token_pos}号token「{token_wanna}」 的张量形状：", tensor_wanna.shape) # 张量类型是float16[0..3583]
  # print("张量头部值是：", tensor_wanna[:10])
  cos_sim = FUNC.cosine_similarity(tensor_embedding.unsqueeze(0), tensor_wanna.unsqueeze(0))
  print("\n{case:",debug_case_traceid, ", word:\"",debug_case_word,"\", debug_case_note:\"",debug_case_note,"\", embeddingTop8:",tensor_embedding[:8].detach().cpu().tolist(), ", cos_sim:",cos_sim.item(),"},")
