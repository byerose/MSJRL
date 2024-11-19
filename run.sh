# python shuffle.py
datasets=("advbench")
shots_values=(256 64 32 16 8 4 2 1)
model_names=("Llama-3.1-8B-Instruct")

for dataset in "${datasets[@]}"; do
    for model_name in "${model_names[@]}"; do
        # 1.jailbreak
        for shots in "${shots_values[@]}"; do
            echo "Running with shots = $shots"
            batch_size=4
            CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
                --model_name $model_name \
                --gpus 4 \
                --batch_size $batch_size \
                --shots $shots \
                --dataset $dataset \
                > logs/${dataset}/${model_name}_${shots}.log 
            batch_size=$((batch_size*2))
        done
        # 2.judge
        CUDA_VISIBLE_DEVICES=4,5,6,7 python judge.py --model_name $model_name --dataset $dataset
        
    done
    reward=1
    for model_name in "${model_names[@]}"; do
        # 1.jailbreak
        for shots in "${shots_values[@]}"; do
            echo "Running with shots = $shots"
            batch_size=4
            CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
                --model_name $model_name \
                --gpus 4 \
                --batch_size $batch_size \
                --shots $shots \
                --reward $reward \
                --dataset $dataset \
                > logs/${dataset}/rl_${model_name}_${shots}.log 
            batch_size=$((batch_size*2))
        done
        # 2.judge
        CUDA_VISIBLE_DEVICES=4,5,6,7 python judge.py --model_name $model_name --reward $reward --dataset $dataset
        
    done
done
