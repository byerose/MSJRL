import pandas as pd
import ast

for shot in [4, 16, 64, 256]:
    for reward in [0]:
        fix = "rl_" if reward == 1 else ""
        df = pd.read_csv(
            f"result_chat/advbench/multiturn_{fix}Llama-3.1-70B-Instruct_{shot}shot.csv"
        )

        counts = {i: {"safe": 0, "unsafe": 0} for i in range(1, 11)}

        for judgement_str in df["judgement"]:
            try:
                judgement_list = ast.literal_eval(judgement_str)
                for i in range(min(10, len(judgement_list))):
                    if judgement_list[i] == "safe":
                        counts[i + 1]["safe"] += 1
                    elif judgement_list[i] == "unsafe":
                        counts[i + 1]["unsafe"] += 1
            except (ValueError, SyntaxError):
                print(f"Error parsing judgement string: {judgement_str}")
        print("reward:", bool(reward), "shot:", shot)
        for i in range(1, 11):
            print(
                f"Position {i}: safe:unsafe = {counts[i]['safe']}:{counts[i]['unsafe']}"
            )
