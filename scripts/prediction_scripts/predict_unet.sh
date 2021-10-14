
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model unet --model_path /root/weights/weights/unet/BERLIN_0806_14:25_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.ckpt --output_folder raw_predictions --domain_adaptation none
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model unet --model_path /root/weights/weights/unet/BERLIN_0806_14:25_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model unet --model_path /root/weights/weights/unet/BERLIN_0806_14:25_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.ckpt --output_folder raw_predictions --domain_adaptation mean_overall


python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model unet --model_path /root/weights/weights/unet/CHICAGO_0805_00:38_vanilla_unet_mse_best_val_loss_2019=42.6634.ckpt --output_folder raw_predictions --domain_adaptation none
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model unet --model_path /root/weights/weights/unet/CHICAGO_0805_00:38_vanilla_unet_mse_best_val_loss_2019=42.6634.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model unet --model_path /root/weights/weights/unet/CHICAGO_0805_00:38_vanilla_unet_mse_best_val_loss_2019=42.6634.ckpt --output_folder raw_predictions --domain_adaptation mean_overall

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model unet --model_path /root/weights/weights/unet/ISTANBUL_0805_23:17_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.ckpt --output_folder raw_predictions --domain_adaptation none
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model unet --model_path /root/weights/weights/unet/ISTANBUL_0805_23:17_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model unet --model_path /root/weights/weights/unet/ISTANBUL_0805_23:17_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.ckpt --output_folder raw_predictions --domain_adaptation mean_overall

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model unet --model_path /root/weights/weights/unet/MELBOURNE_0804_19:42_vanilla_unet_mse_best_val_loss_2019=26.7588.ckpt --output_folder raw_predictions --domain_adaptation none
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model unet --model_path /root/weights/weights/unet/MELBOURNE_0804_19:42_vanilla_unet_mse_best_val_loss_2019=26.7588.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model unet --model_path /root/weights/weights/unet/MELBOURNE_0804_19:42_vanilla_unet_mse_best_val_loss_2019=26.7588.ckpt --output_folder raw_predictions --domain_adaptation mean_overall





