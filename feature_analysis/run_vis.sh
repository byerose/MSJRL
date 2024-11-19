#rewards=(0 1)
#shot_values=(256 64 16 4 0)
# for reward in "${rewards[@]}"; do
#     for shot in "${shot_values[@]}"; do
#         python get_response_activations.py --shot "$shot" --reward "$reward"
#     done
# done

rewards=(0 1)
shot_values=(256 64 16 4)
for reward in "${rewards[@]}"; do
    for shot in "${shot_values[@]}"; do
        python visualization.py --shot "$shot" --reward "$reward"
    done
done
