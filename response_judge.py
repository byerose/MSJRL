import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = "Llama-Guard-3-8B"
model_path = os.path.join("/disk/mount/models/", model_id)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
device = "cuda:2"


def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


file_path = f"data/many_responses.csv"
with open(file_path, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    col_names = ["Question", "Response", "Judge", "Category"]
    with open(
        f"data/many_responses_judge.csv", "w", newline="", encoding="utf-8"
    ) as new_csvfile:
        writer = csv.DictWriter(new_csvfile, fieldnames=col_names)
        writer.writeheader()  # 写入表头
        for row in reader:
            behavior = row["Question"]
            response = row["Response"]
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
                        ]
                    )
                )
                .strip("\n")
                .split("\n")
            )
            print(repr(judgment))
            judge = judgment[0]
            if len(judgment) > 1:
                category = judgment[1]
            else:
                category = " "
            writer.writerow(
                {
                    "Question": behavior,
                    "Response": response,
                    "Judge": judge,
                    "Category": category,
                }
            )
print("done.")
