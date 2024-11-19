import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import argparse, os

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--dataset", type=str, default="advbench", required=True)
parser.add_argument("--multiturn", type=bool, default=False)
args = parser.parse_args()
if args.multiturn:
    prefix = "multiturn_"
else:
    prefix = ""
model_data = {}
for model_id in [
    "Llama-3.1-70B-Instruct",
    "Qwen2.5-72B-Instruct",
    # "Llama-2-70b-chat-hf",
    # "Llama-3.1-8B-Instruct",
    # "Qwen2.5-7B-Instruct",
]:
    for method in ["no reward", "naive reward"]:
        # Initialize model_data[method] if not already initialized
        if method not in model_data:
            model_data[method] = {}
        # Initialize model_data[method][model_id] if not already initialized
        if model_id not in model_data[method]:
            model_data[method][model_id] = []
        for shot in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if method == "naive reward":
                file_path = (
                    f"result_judge/{args.dataset}/{prefix}rl_{model_id}_{shot}shot.csv"
                )
            else:
                file_path = (
                    f"result_judge/{args.dataset}/{prefix}{model_id}_{shot}shot.csv"
                )
            if not os.path.exists(file_path):
                continue
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                total_responses = 0
                harm_responses = 0
                for row in reader:
                    response = row["response"]
                    total_responses += 1
                    if row["judge"] == "unsafe":
                        harm_responses += 1
                # 计算不以"I can't"开头的response的比例
                if total_responses > 0:
                    proportion = harm_responses / total_responses
                else:
                    proportion = 0
                model_data[method][model_id].append({shot: proportion})
                # 输出结果
                print(f"{file_path}: {proportion:.2%} succeed.")

print(model_data)
# Prepare the header for the CSV file
csv_header = ["number of shots"]
# Specify the output CSV file name
output_file_path = f"result_draw/{args.dataset}/{prefix}harmful_responses_summary.csv"
# Open the output file in write mode
with open(output_file_path, mode="w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    for method in ["no reward", "naive reward"]:
        for model_id, data in model_data[method].items():
            csv_header.append(model_id + ":" + method)
    # Write the header to the CSV file
    writer.writerow(csv_header)

    d = []

    for shot in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        # Iterate over each model and its data
        d.clear()
        d.append(shot)
        for method in ["no reward", "naive reward"]:
            for model_id, data in model_data[method].items():
                # Iterate over each shot-proportion pair
                for i in data:
                    if shot in i:
                        d.append(i[shot])
        writer.writerow(d)

print(f"Data successfully saved to {output_file_path}")
shots = []
proportions = []
for method in ["no reward", "naive reward"]:
    # 绘制折线图
    for model_id, data in model_data[method].items():
        if len(data) < 1:
            continue
        shots.clear()
        proportions.clear()
        for d in data:
            shots.extend(list(d.keys()))
            proportions.extend(list(d.values()))
        plt.plot(shots, proportions, marker="o", label=model_id)

    # 添加图表标题和标签
    plt.title("Malicious Use Cases")
    plt.xlabel("number of shots")
    plt.ylabel("% of harmful responses")
    plt.xscale("log", base=2)

    # 设置纵轴为百分数格式
    def to_percent(y, pos):
        return f"{100 * y:.0f}%"  # 将小数转换为百分数

    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)  # 应用格式化器到y轴
    plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])  # 设置x轴刻度为每个样本数
    plt.ylim(0, 1)  # y轴范围从0到1
    plt.grid()
    plt.legend()
    # 保存图表到本地
    if method == "naive reward":
        save_fig = f"result_draw/{args.dataset}/{prefix}all_models_rl.png"
    else:
        save_fig = f"result_draw/{args.dataset}/{prefix}all_models.png"
    plt.savefig(
        save_fig,
        dpi=300,
        bbox_inches="tight",
    )  # 保存为PNG文件
    plt.close()  # 关闭图形以释放内存
