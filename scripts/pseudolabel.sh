NUM_ROUNDS=1
MAX_EPOCHS=5

DATASET_PATH="/home/trcomp/data/traffic4cast"
BATCH_SIZE=4
NUM_WORKERS=4
LR=0.0001
DEVICE="cuda"
NET="vanilla_unet"

MELBOURNE_WEIGHTS="/home/trcomp/repos/weights_submission/weights/MELBOURNE_0804_19:42_vanilla_unet_mse_best_val_loss_2019=26.7588.ckpt"
CHICAGO_WEIGHTS="/home/trcomp/repos/weights_submission/weights/CHICAGO_0805_00:38_vanilla_unet_mse_best_val_loss_2019=42.6634.ckpt"
ISTANBUL_WEIGHTS="/home/trcomp/repos/weights_submission/weights/ISTANBUL_0805_23:17_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.ckpt"
BERLIN_WEIGHTS="/home/trcomp/repos/weights_submission/weights/BERLIN_0806_14:25_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.ckpt"

# MELBOURNE 

python pseudolabel.py --dataset_path $DATASET_PATH --city MELBOURNE --checkpoint_path $MELBOURNE_WEIGHTS --net $NET --device $DEVICE --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --learning_rate $LR --num_rounds $NUM_ROUNDS --max_epochs $MAX_EPOCHS

# CHICAGO 

python pseudolabel.py --dataset_path $DATASET_PATH --city CHICAGO --checkpoint_path $CHICAGO_WEIGHTS --net $NET --device $DEVICE --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --learning_rate $LR --num_rounds $NUM_ROUNDS --max_epochs $MAX_EPOCHS

# ISTANBUL 

python pseudolabel.py --dataset_path $DATASET_PATH --city ISTANBUL --checkpoint_path $ISTANBUL_WEIGHTS --net $NET --device $DEVICE --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --learning_rate $LR --num_rounds $NUM_ROUNDS --max_epochs $MAX_EPOCHS

# BERLIN 

python pseudolabel.py --dataset_path $DATASET_PATH --city BERLIN --checkpoint_path $BERLIN_WEIGHTS --net $NET --device $DEVICE --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --learning_rate $LR --num_rounds $NUM_ROUNDS --max_epochs $MAX_EPOCHS
