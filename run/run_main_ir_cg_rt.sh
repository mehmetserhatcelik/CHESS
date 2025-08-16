source .env
data_mode='dev'
data_path='./data/dev/dev.json'

config="./run/configs/CHESS_IR_CG_RT.yaml"

num_workers=1

python3 -u ./src/main.py --data_mode ${data_mode} --data_path ${data_path} --config "$config" \
        --num_workers ${num_workers} --pick_final_sql true
