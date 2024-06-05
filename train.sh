python3 train.py 
--dataset oxford_pets \
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
