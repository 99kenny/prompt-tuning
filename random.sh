# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 100 --exp imgp_no_batchwise_2
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 10 --exp imgp_no_batchwise_3
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 20 --exp imgp_no_batchwise_4
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 30 --exp imgp_no_batchwise_5
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 40 --exp imgp_no_batchwise_6
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 50 --exp imgp_no_batchwise_7
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 60 --exp imgp_no_batchwise_8
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 70 --exp imgp_no_batchwise_9
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 80 --exp imgp_no_batchwise_10
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 90 --exp imgp_no_batchwise_11
# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 110 --exp imgp_no_batchwise_12


# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_imgp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 42
CUDA_VISIBLE_DEVICES=4 python main.py cifar100_genp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 42

CUDA_VISIBLE_DEVICES=4 python main.py cifar100_ptchp --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 42

# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_ptchp_s --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 100 --exp ptchp_s 

# CUDA_VISIBLE_DEVICES=4 python main.py cifar100_ptchp_s --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output --seed 100 --exp ptchp_s_2_deepinversion --var --l2