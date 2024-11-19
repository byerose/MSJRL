import argparse
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--reward", type=bool, default=False)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset", type=str, default="advbench", required=True)
parser.add_argument("--multiturn", type=int, default=0)
args = parser.parse_args()

model_id = "Llama-Guard-3-8B"
model_path = os.path.join("/disk/mount/models/", model_id)
device = "cuda:2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


def moderate(chat, device):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


if args.multiturn == 1:
    prefix = "multiturn_"
else:
    prefix = ""
model_data = {}
model_id = args.model_name
model_data[model_id] = []
for shot in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    if args.reward:
        file_path = f"result_chat/{args.dataset}/{prefix}rl_{model_id}_{shot}shot.csv"
    else:
        file_path = f"result_chat/{args.dataset}/{prefix}{model_id}_{shot}shot.csv"
    print(file_path)
    if not os.path.exists(file_path):
        continue
    if args.reward:
        file_path_judge = (
            f"result_judge/{args.dataset}/{prefix}rl_{model_id}_{shot}shot.csv"
        )
    else:
        file_path_judge = (
            f"result_judge/{args.dataset}/{prefix}{model_id}_{shot}shot.csv"
        )
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        col_names = reader.fieldnames + ["judge", "category"]
        with open(file_path_judge, "w", newline="", encoding="utf-8") as new_csvfile:
            writer = csv.DictWriter(new_csvfile, fieldnames=col_names)
            writer.writeheader()
            for row in reader:
                behavior = row["behavior"]
                response = row["response"]
                judgment = (
                    (
                        moderate(
                            [
                                {
                                    "role": "user",
                                    "content": behavior,
                                },
                                {
                                    "role": "assistant",
                                    "content": response,
                                },
                            ],
                            device,
                        )
                    )
                    .strip("\n")
                    .split("\n")
                )
                print(repr(judgment))
                row["judge"] = judgment[0]
                if len(judgment) > 1:
                    row["category"] = judgment[1]
                else:
                    row["category"] = " "
                writer.writerow(row)
print("done.")
