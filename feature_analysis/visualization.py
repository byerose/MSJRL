import argparse
import pandas as pd
import sys

from unsupervised import *
from supervised import *

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--shot", type=int, required=True)
parser.add_argument("--reward", type=int, required=True)
args = parser.parse_args()
reward = "rl_" if args.reward == True else ""
shot = args.shot
path_attack = f"./result/layer31_{reward}advbench_{shot}shot_res_activations.pkl"
df = pd.read_pickle(path_attack)
print(df.columns.values)
## CONFIG
model_name = "Llama-3.1-8B-Instruct"
layer = 31
activation_column = "prompt_res_activations"
dataset_name = "advbench"

df = df.rename(
    columns={
        "behavior": "jailbreak_prompt_name",
        "res_activations": "prompt_res_activations",
    }
)

# 对数据框进行条件筛选和更新：
# 1. 使用str.contains()检查"jailbreak_prompt_name"列中是否包含"benign"字符串
# 2. na=False 参数表示忽略空值(NA/NaN)
# 3. 对于满足条件的行，将"rating"列的值设置为"No"
num_samples = len(df)
unsafe_ratio = len(df[df["judge"] == "unsafe"]) / num_samples
safe_ratio = len(df[df["judge"] == "safe"]) / num_samples

print(f"Unsafe比例: {unsafe_ratio:.2%}")
print(f"Safe比例: {safe_ratio:.2%}")

# 创建一个空列表用于存储成功的越狱提示名称


# ## Unsupervised Learning Methods

# 不是连接所有token嵌入,而是对每个序列取平均嵌入
vectors = [act.squeeze(0) for act in df[activation_column]]
# 对每个张量在序列长度维度(dim=0)上取平均
flattened = [act.mean(dim=0).cpu().numpy() for act in vectors]
# 堆叠所有平均嵌入
tensors_flat = np.stack(flattened, axis=0)
print(f"tensors_flat 的形状: {tensors_flat.shape}")
print(f"标签数量: {len(df['judge'])}")
assert (
    len(df["judge"]) == tensors_flat.shape[0]
), "Number of labels must match number of samples"
visualize_act_with_t_sne(
    model_name=model_name,
    layer=layer,
    tensors_flat=tensors_flat,
    labels=df["judge"],
    dataset_name=dataset_name,
    num_samples=num_samples,
    perplexity=20,
    scores=None,
    save=True,
    shot=shot,
    reward=reward,
)


# plt = visualize_act_with_t_sne(
#     model_name=model_name,
#     layer=layer,
#     tensors_flat=tensors_flat,
#     labels=df["jailbreak_prompt_name"],
#     dataset_name=dataset_name,
#     num_samples=num_samples,
#     perplexity=9,
#     scores=None,
#     save=True,
# )


# visualize_act_with_t_sne_interactive(
#     model_name=model_name,
#     layer=18,
#     tensors_flat=tensors_flat,
#     labels=df["prompt_name"],
#     dataset_name=dataset_name,
#     num_samples=num_samples,
#     perplexity=9,
#     scores=None,
#     save=True,
# )


# # Visualize with PCA


# visualize_act_with_pca(
#     model_name=model_name,
#     layer=18,
#     tensors_flat=tensors_flat,
#     labels=df["judge"],
#     dataset_name=dataset_name,
#     num_samples=num_samples,
#     scores=None,
#     save=True,
#     shot=shot,
#     reward=reward,
# )


# visualize_act_with_pca(
#     model_name=model_name,
#     layer=layer,
#     tensors_flat=tensors_flat,
#     labels=df["jailbreak_prompt_name"],
#     dataset_name=dataset_name,
#     num_samples=num_samples,
#     scores=None,
#     save=True,
# )
