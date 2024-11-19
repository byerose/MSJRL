python shuffle.py
datasets=("advbench")
shots_values=(4 16 64 256)
model_names=("Llama-3.1-70B-Instruct")
for dataset in "${datasets[@]}"; do
    for model_name in "${model_names[@]}"; do
        # 1.jailbreak
        for shots in "${shots_values[@]}"; do
            echo "Running with shots = $shots"
            
            CUDA_VISIBLE_DEVICES=4,5,6,7 python main_multiturn.py \
                --model_name $model_name \
                --gpus 4 \
                --turns 10 \
                --shots $shots \
                --dataset $dataset \
                > logs/${dataset}/multi_${model_name}_${shots}.log 
        done
        # 2.judge
        # CUDA_VISIBLE_DEVICES=4,5,6,7 python judge.py --model_name $model_name --dataset $dataset --multiturn 1
        
    done
    
    reward=1
    for model_name in "${model_names[@]}"; do
        # 1.jailbreak
        for shots in "${shots_values[@]}"; do
            echo "Running with shots = $shots"
            CUDA_VISIBLE_DEVICES=4,5,6,7 python main_multiturn.py \
                --model_name $model_name \
                --gpus 4 \
                --turns 10 \
                --shots $shots \
                --reward $reward \
                --dataset $dataset \
                > logs/${dataset}/multi_rl_${model_name}_${shots}.log 
        done
        # 2.judge
        # CUDA_VISIBLE_DEVICES=4,5,6,7 python judge.py --model_name $model_name --reward $reward --dataset $dataset --multiturn 1
        
    done
done
