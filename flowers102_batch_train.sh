python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type full_ft \
--use_hooks hooks_req \
--block 11 \
--tokens_used mean \
--device cuda:0

python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type linear_probing \
--use_hooks hooks_req \
--block 11 \
--tokens_used mean \
--device cuda:1

python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type full_ft \
--use_hooks hooks_req \
--block 11 \
--tokens_used top_one \
--device cuda:2

python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type linear_probing \
--use_hooks hooks_req \
--block 11 \
--tokens_used top_one \
--device cuda:3

python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type full_ft \
--use_hooks hooks_req \
--block 11 \
--tokens_used cls \
--device cuda:4

python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type linear_probing \
--use_hooks hooks_req \
--block 11 \
--tokens_used cls \
--device cuda:5

python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type full_ft \
--use_hooks no_hooks \
--device cuda:6

python3 train.py 
--dataset flowers_102 \
--data_dir $1 \
--save_dir $2 \
--train_type linear_probing \
--use_hooks no_hooks \
--device cuda:7