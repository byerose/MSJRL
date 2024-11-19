import pandas as pd

# 文件名列表
filenames = [
    f"../result_judge/advbench/Llama-3.1-8B-Instruct_{i}shot.csv"
    for i in [4, 16, 64, 256]
]
filenames.extend(
    [
        f"../result_judge/advbench/rl_Llama-3.1-8B-Instruct_{i}shot.csv"
        for i in [4, 16, 64, 256]
    ]
)
# 读取所有文件到DataFrame列表
dataframes = [pd.read_csv(filename) for filename in filenames]

# 获取8shot.csv中judge为'unsafe'的行的索引
unsafe_indices = dataframes[-1][dataframes[-1]["judge"] == "unsafe"].index

# 输出其他文件中对应行的judge值
for i, df in enumerate(dataframes[:]):  # 忽略最后一个即8shot.csv
    print(f"Judge values in {filenames[i]} for unsafe indices in 8shot.csv:")
    for idx in unsafe_indices:
        if idx < len(df):
            print(f"Index {idx}: {df.at[idx, 'judge']}")
        else:
            print(f"Index {idx} is out of bounds for {filenames[i]}")
    print()

# 筛选judge列为'unsafe'的行
unsafe_rows = dataframes[-1][dataframes[-1]["judge"] == "unsafe"]

# 将筛选后的行保存为新的CSV文件
unsafe_rows.to_csv("data.csv", index=False)
