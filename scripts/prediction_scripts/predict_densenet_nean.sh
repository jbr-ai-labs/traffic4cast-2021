
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model densenet_unet --model_path /root/weights/weights/densenet/BERLIN_1008_14:30_densenet_unet_mse_best_val_loss_2019=78.4303.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model densenet_unet --model_path /root/weights/weights/densenet/CHICAGO_1010_17:30_densenet_unet_mse_best_val_loss_2019=41.1579.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model densenet_unet --model_path /root/weights/weights/densenet/MELBOURNE_1009_16:19_densenet_unet_mse_best_val_loss_2019=25.7395.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model densenet_unet --model_path /root/weights/weights/densenet/ISTANBUL_1011_11:39_densenet_unet_mse_best_val_loss_2019=nan_v1.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ; 


