python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type full_ft \
--use_hooks hooks_req \
--block 11 \
--tokens_used mean \
--num_tokens_for_mean 10 \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:0

python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type linear_probing \
--use_hooks hooks_req \
--block 11 \
--tokens_used mean \
--num_tokens_for_mean 10 \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:1

python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type full_ft \
--use_hooks hooks_req \
--block 11 \
--tokens_used top_one \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:2

python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type linear_probing \
--use_hooks hooks_req \
--block 11 \
--tokens_used top_one \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:3

python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type full_ft \
--use_hooks hooks_req \
--block 11 \
--tokens_used cls \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:4

python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type linear_probing \
--use_hooks hooks_req \
--block 11 \
--tokens_used cls \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:5

python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type full_ft \
--use_hooks no_hooks \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:6

python3 train.py 
--dataset flowers_102 \
--data_dir /mnt/d/ViT \
--download_dataset True \
--save_dir /mnt/d/ViT \
--train_type linear_probing \
--use_hooks no_hooks \
--train_batch_size 32 \
--test_batch_size 32 \
--num_workers 2 \
--learning_rate 5e-6 \
--weight_decay 1e-6 \
--device cuda:7