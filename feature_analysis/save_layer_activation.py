import pandas as pd

for reward in [0]:
    prefix = "rl_" if reward == 1 else ""
    for shot in [4]:
        # Paths for the original and attack DataFrames
        path = "./result/advbench_0shot_res_activations.pkl"
        path_attack = f"./result/{prefix}advbench_{shot}shot_res_activations.pkl"

        # Load the DataFrames
        df1 = pd.read_pickle(path)
        df2 = pd.read_pickle(path_attack)

        # Concatenate the DataFrames
        df = pd.concat([df1, df2], axis=0)

        # Extract tensors from the DataFrame
        tensors = [row[31][0] for row in df["res_activations"]]

        # Update the "res_activations" column with the new tensors
        df["res_activations"] = tensors

        # Construct the new file path to save the updated DataFrame
        new_file_path = (
            f"./result/layer31_{prefix}advbench_{shot}shot_res_activations.pkl"
        )

        # Save the updated DataFrame to a new pickle file
        df.to_pickle(new_file_path)

        print(f"Saved updated DataFrame to {new_file_path}")
