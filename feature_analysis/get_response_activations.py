import os
import pickle
import pandas as pd
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--shot", type=int, required=True)
parser.add_argument("--reward", type=int, required=True)
args = parser.parse_args()

quantization_config = BitsAndBytesConfig(
    # load_in_8bit=True
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def get_activations_from_df(
    model,
    tokenizer,  # 模型名称
    many_prompt,
    shot,
    reward,
    df,  # 输入的数据框
    prompt_column,  # 包含提示文本的列名
    n_rows=None,  # 要处理的行数，默认为None表示处理所有行
    n_start=0,  # 起始行号
    dataset_name="advbench",  # 数据集名称
    token_position="all",  # token位置，可以是"all"或特定位置
    generate_output=False,  # 是否生成输出文本
    type_of_activations="res_activations",  # 激活值类型
):
    """
    从数据框中获取模型对指定文本的激活值

    参数说明：
    - model_name: 使用的模型名称
    - df: 包含输入文本的数据框
    - prompt_column: 输入文本所在的列名
    - n_rows: 处理的行数，None表示处理所有行
    - n_start: 开始处理的行号
    - dataset_name: 数据集名称，用于保存结果
    - token_position: 获取激活值的token位置
    - generate_output: 是否生成模型输出
    - type_of_activations: 激活值类型
    """

    # 加载模型和分词器

    print(f"Dataset loaded successfully with {n_rows} rows")

    # 初始化激活值列
    df[type_of_activations] = None

    counter = 0

    # 如果未指定处理行数，则处理所有行
    if n_rows == None:
        n_rows = len(df)

    # 遍历数据框中的每一行
    for row in df.iloc[:n_rows].iterrows():
        counter += 1
        index = row[0]
        print(f"Processing prompt {counter}/{n_rows}")

        # 获取当前行的提示文本
        prompt = many_prompt + row[1][prompt_column]

        # 获取模型对该提示文本的激活值
        output_activations = get_res_activations(
            model=model,
            tokenizer=tokenizer,
            token_position=token_position,
            prompt=prompt,
            type_of_activations=type_of_activations,
            generate_output=generate_output,
        )

        # 保存激活值到数据框中
        df.at[index, type_of_activations] = output_activations[type_of_activations]

        # 如果需要生成输出，保存输出文本
        if generate_output:
            df.at[index, "output_text"] = output_activations["output_text"]

    print("Processing complete")

    # 保存处理后的数据框并返回文件路径
    return save_with_timestamp(
        shot=shot,
        reward=reward,
        dataset_name=dataset_name,
        save=True,
        df=df,
        add_info=type_of_activations,  # 添加激活值类型信息到文件名
    )


def save_with_timestamp(
    dataset_name,
    shot,
    reward,
    save=False,
    df=None,
    add_info="",
):
    """
    将数据框保存为带时间戳的pickle文件

    参数：
    - model_name: 模型名称
    - dataset_name: 数据集名称
    - save: 是否保存文件的标志，默认为False
    - df: 要保存的数据框
    - add_info: 文件名中的额外信息

    返回：
    - save_path: 保存文件的完整路径
    """

    # 获取当前时间戳，格式为"日月年时:分:秒"

    # 处理模型名称和数据集名称，移除路径分隔符"/"
    dataset_name = dataset_name.replace("/", "")

    # 设置保存目录
    save_dir = "./result"

    # 如果保存目录不存在，创建该目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 构建保存文件的完整路径
    # 文件名格式：数据集名称_模型名称_额外信息_时间戳.pkl
    save_path = f"{save_dir}/{reward}{dataset_name}_{shot}shot_{add_info}.pkl"

    # 如果save为True，将数据框保存为pickle文件
    if save:
        with open(save_path, "wb") as f:
            pickle.dump(df, f, protocol=3)

    # 返回文件保存路径
    return save_path


def get_model(model_name: str):
    """
    加载预训练语言模型和对应的分词器

    Args:
        model_name (str): 模型名称或路径

    Returns:
        tuple: 包含以下两个元素:
            - model: 加载的语言模型
            - tokenizer: 对应的分词器
    """
    # 加载预训练模型
    # trust_remote_code=True 允许从远程加载代码
    # device_map="auto" 自动处理模型在设备间的分配
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="balanced",
        max_memory={
            4: "80GiB",
            5: "80GiB",
            6: "80GiB",
            7: "80GiB",
            "cpu": "80GiB",
        },
        torch_dtype=torch.float16,
        # quantization_config=quantization_config,
    )

    # 加载对应的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 设置填充token为结束token
    # 这样做是为了确保在处理变长序列时能正确进行填充
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_res_layers_to_enumerate(model):
    """
    获取不同模型架构的层结构

    Args:
        model: 预训练语言模型实例

    Returns:
        模型的层结构序列

    Raises:
        ValueError: 当输入的模型类型不受支持时抛出异常
    """
    # 获取模型名称
    model_name = model.config._name_or_path

    # 根据模型名称返回对应的层结构
    # GPT系列模型
    if "gpt" in model_name:
        return model.transformer.h

    # Pythia系列模型
    elif "pythia" in model_name:
        return model.gpt_neox.layers

    # BERT系列模型
    elif "bert" in model_name:
        return model.encoder.layer

    # Mistral系列模型
    elif "Mistral" in model_name:
        return model.model.layers

    # Gemma系列模型
    elif "gemma" in model_name:
        return model.model.layers

    # LLaMA系列模型
    elif "llama" in model_name.lower():
        return model.model.layers

    # 不支持的模型类型
    else:
        raise ValueError(f"Unsupported model: {model_name}.")


def get_res_activations(
    model,
    tokenizer,
    prompt,
    type_of_activations="res_activations",
    token_position="all",
    generate_output=False,
):
    """
    获取模型中间层的激活值

    Args:
        model: 预训练语言模型
        tokenizer: 分词器
        prompt: 输入文本
        type_of_activations: 激活值类型的标识符
        token_position: 需要获取的token位置 ("last"或"all")
        generate_output: 是否生成模型输出文本

    Returns:
        dict: 包含激活值和可选的生成文本
        激活值维度: [batch_size, sequence_length, num_hidden_layers, hidden_size]
    """
    # 获取模型所在设备
    device = next(model.parameters()).device

    # 将输入文本转换为token ID
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    len_prompt = len(input_ids)

    # 存储激活值的字典
    activations = {}

    def get_activation(name):
        """
        创建用于获取激活值的钩子函数

        Args:
            name: 层的标识符
        """

        def hook(model, input, output):
            if name not in activations:
                activations[name] = []

            # 根据token_position参数获取不同位置的激活值
            if token_position == "all":
                # 获取所有token位置的激活值
                activations[name].append(output[0][:, :, :].detach().cpu())
            elif token_position == "last":
                # 只获取最后一个token位置的激活值
                activations[name].append(output[0][:, -1, :].detach().cpu())
            else:
                raise ValueError(f"Unsupported token_position: {token_position}")

        return hook

    # 获取需要遍历的模型层
    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # 为每一层注册前向传播钩子
    for i, layer in enumerate(layers_to_enum):
        hook_handle = layer.register_forward_hook(get_activation(i))
        hooks.append(hook_handle)

    # 运行模型前向传播
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        _ = model(**inputs)

    # 移除所有钩子
    for h in hooks:
        h.remove()

    print("Activations shape:", activations[0][0].shape)

    if generate_output:
        # 生成模型输出文本
        output_sequence = model.generate(
            input_ids,
            num_return_sequences=1,  # 生成序列的数量
            max_new_tokens=200,  # 最大新生成的token数量
            do_sample=False,  # 是否使用采样策略
        )

        # 解码生成的文本
        generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)

        # 移除生成文本中的原始提示部分
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        return {"output_text": generated_text, type_of_activations: activations}

    return {type_of_activations: activations}


def average_across_sequence(activation):
    """
    计算激活值在序列维度上的平均值

    Args:
        activation: 输入的激活值张量
            预期维度: [batch_size, sequence_length, hidden_size]

    Returns:
        torch.Tensor: 序列维度平均后的激活值
            输出维度: [batch_size, hidden_size]
    """
    # 在维度1（序列长度维度）上计算平均值
    # dim=1 表示在第二个维度（序列长度）上进行平均
    return activation.mean(dim=1)


model_name = "Llama-3.1-8B-Instruct"
model_path = f"/disk/mount/models/{model_name}"
model, tokenizer = get_model(model_path)
shot = args.shot
reward = "rl_" if args.reward == 1 else ""
print("reward:", reward)
with open(
    "../data/balanced_many_responses_judge.csv",
    newline="",
    encoding="utf-8",
) as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)
    selected_rows = rows[:shot]
    many_prompt = ""
    for row in selected_rows:
        many_prompt += row["Question"]
        many_prompt += "\n"
        many_prompt += row["Response"]
        many_prompt += "\n"
        if reward == "rl_":
            many_prompt += str("Your answer is absolutely correct! Fantastic work!")
            many_prompt += "\n"

csv_file_name = f"../result_judge/advbench/{reward}{model_name}_{shot}shot.csv"
ex_file = "./data.csv"
ex_df = pd.read_csv(ex_file)
result_df = pd.read_csv(csv_file_name)
# 获取ex_file的前10行的behavior值
ex_behaviors = ex_df.head(20)["behavior"].tolist()
# 从csv_file_name中选择behavior值与ex_behaviors匹配的行
df = result_df[result_df["behavior"].isin(ex_behaviors)]
# print(df.columns)
# 设置模型
# 获取数据框的行数作为样本数
num_samples = len(df)
# 设置数据集名称
dataset_name = "advbench"
# 调用get_activations_from_df函数获取模型激活值
# 参数说明：
# - model_name: 使用的模型名称
# - prompt_column: 指定"forbidden_prompt"列作为输入文本
# - df: 输入的数据框
# - n_rows: 处理的行数（这里处理所有行）
save_path = get_activations_from_df(
    model=model,
    tokenizer=tokenizer,
    many_prompt=many_prompt,
    shot=shot,
    reward=reward,
    prompt_column="behavior",
    dataset_name=dataset_name,
    df=df,
    n_rows=num_samples,
)

# 打印保存文件的路径
print(save_path)
