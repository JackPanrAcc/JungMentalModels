# Author: jackpanr
# Timely: 2026-10-06
# Descript: 
#   计算大模型“ffn降维清算操作”的操作之前的《就绪层的层号》，和操作后《维度数趋稳时的层号》
#   严肃算法应该代入提示词之后，逐层计算“Intrinsic Dimensionality 本征内在维度”，用二次导数确定上述层号 https://arxiv.org/html/2501.10573v2
#
#   本脚本中简化后使用 up project erank 确定上述层号
#     就绪层层号maxuperank_layer_number = uperank最大值时的层号
#     趋稳层层号introspect_layer_number = 降维清算启动之后、uperank二阶导数为零时的层号
#     考虑到不恰当的训练中qkv可能造成维数波动，等工程原因。实际使用的趋稳层层号可以再额外加一
#


from transformers import AutoModelForCausalLM
import math
import torch
import os
import gc


# 目标大模型
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 限制底层线程数防止多现场假死，增加稳定性以便放后台自行跑数据
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


# 辅助函数：计算有效秩erank
def calculate_erank(tensor: torch.Tensor) -> float:
    # 因为intel 芯片macbook 只有CPU 平台所以这里是强制转到 CPU 并清理梯度。使用时请根据实际情况切换环境
    matrix = tensor.detach().float().cpu()
    if matrix.ndim > 2:
        matrix = matrix.view(matrix.size(0), -1)

    # driver='gesvd' 速度虽然比默认的 gesdd 慢一点，但它对 Intel CPU 内存管理稳定
    _, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    sum_s = torch.sum(s)
    if sum_s == 0:
        return 0.0
    p = s / sum_s
    p = p[p > 0]

    entropy = -torch.sum(p * torch.log(p))
    return torch.exp(entropy).item()



# =============== 逐层轮询主逻辑 ===============
# 1. 安全加载模型
print("正在从本地或网络加载模型，请稍候...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
print(f"\ntype of model is: {model.config.torch_dtype}") # 提示from_pretrained 执行完成

layers = model.model.layers
num_layers = len(layers)


# 2. 串行收集所有层的 eRank 数据
gate_eranks = []
up_eranks = []
down_eranks = []
print("\n正在计算各层矩阵的有效秩...")
for i, layer in enumerate(layers):
    gate_eranks.append(calculate_erank(layer.mlp.gate_proj.weight))
    up_eranks.append(calculate_erank(layer.mlp.up_proj.weight))
    down_eranks.append(calculate_erank(layer.mlp.down_proj.weight))
    print(f"\nLayer {i} 计算完成. up_proj.weight is: {layer.mlp.up_proj.weight}") # 提示目前正在执行第一层的有效秩
    gc.collect() # 算完一层立即强行回收垃圾，如果算力资源足够可忽略


# 3. 计算 up project erank 的一阶导数和二阶导数
d1_up = [0.0] * num_layers
d2_up = [0.0] * num_layers
for i in range(num_layers):
    # 一阶导数
    if i == 0:
        d1_up[i] = up_eranks[i+1] - up_eranks[i]
    elif i == num_layers - 1:
        d1_up[i] = up_eranks[i] - up_eranks[i-1]
    else:
        d1_up[i] = (up_eranks[i+1] - up_eranks[i-1]) / 2.0

    # 二阶导数
    if i == 0:
        d2_up[i] = up_eranks[i+2] - 2 * up_eranks[i+1] + up_eranks[i]
    elif i == num_layers - 1:
        d2_up[i] = up_eranks[i] - 2 * up_eranks[i-1] + up_eranks[i-2]
    else:
        d2_up[i] = up_eranks[i+1] - 2 * up_eranks[i] + up_eranks[i-1]


# 4. 打印总数据表格
header = f"{'Layer':<6} | {'gate erank':<12} | {'up erank':<12} | {'down erank':<12} | {'up erank 1stDeriv':<15} | {'up erank 2ndDeriv':<15}"
print("\n" + header)
print("-" * len(header))
for i in range(num_layers):
    print(f"{i:<6} | {gate_eranks[i]:<12.2f} | {up_eranks[i]:<12.2f} | {down_eranks[i]:<12.2f} | {d1_up[i]:<15.4f} | {d2_up[i]:<15.4f}")


# 5. 输出目标层号
print("\n" + "="*50)
print(f"  {model_name} 的降维清算操作相关层如下：")
print("="*50)

# 寻找最大信息熵层（排除最前最后的边界层）
search_range_up = up_eranks[1:-1]
max_up_val = max(search_range_up)
max_up_layer = up_eranks.index(max_up_val)
print(f"推荐就绪层层号maxuperank_layer_number = {max_up_layer} （该层本征内在维度上限 up erank: {max_up_val:.2f}）")

# 在最大熵层之后，寻找降维过程中二阶导数最接近 0 的层
after_collapse_layers = list(range(max_up_layer + 1, num_layers - 1))
if after_collapse_layers:
    zero_d2_layer = min(after_collapse_layers, key=lambda idx: abs(d2_up[idx]))
    
    engineered_stabilized_layer = zero_d2_layer + 1
    # 【逻辑修正】：将末尾的 max_up_val 替换为当前趋稳层实际的有效秩 up_eranks[zero_d2_layer]
    print(f"理论趋稳层层号introspect_layer_number0= {zero_d2_layer} （该层本征内在维度上限 up erank: {up_eranks[zero_d2_layer]:.2f}）")
    print(f"推荐趋稳层层号introspect_layer_number = {engineered_stabilized_layer} （理论趋稳层层号加一，用以规避qkv 工程训练等扰动）")
else:
    print("\n未能在中后段捕捉到趋稳收缩状态")
print("="*50)


