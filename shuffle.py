import csv
import random

# 读取并打乱 CSV 文件内容
with open(
    "data/balanced_many_responses_judge.csv", newline="", encoding="utf-8"
) as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)  # 读取所有行数据
    fieldnames = reader.fieldnames  # 获取 CSV 文件的字段名（列名）

# 打乱列表的顺序
random.shuffle(rows)

# 将打乱的数据写回到原文件
with open(
    "data/balanced_many_responses_judge.csv", mode="w", newline="", encoding="utf-8"
) as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 写入打乱后的数据
    writer.writerows(rows)
