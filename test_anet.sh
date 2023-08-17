python main.py --task activitynet \
            --predictor transformer \
             --mode test \
             --char_dim 100 \
             --gpu_idx 6 \
             --batch_size 64 \
             --force checkpoint/activitynet
             