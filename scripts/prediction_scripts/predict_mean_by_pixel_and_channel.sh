
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model unet --model_path /root/weights/weights/unet/BERLIN_0806_14:25_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model unet --model_path /root/weights/weights/unet/CHICAGO_0805_00:38_vanilla_unet_mse_best_val_loss_2019=42.6634.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model unet --model_path /root/weights/weights/unet/ISTANBUL_0805_23:17_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model unet --model_path /root/weights/weights/unet/MELBOURNE_0804_19:42_vanilla_unet_mse_best_val_loss_2019=26.7588.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel


python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model effnetb5_unet --model_path /root/weights/weights/effnetb5/BERLIN_1008_14:30_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model effnetb5_unet --model_path /root/weights/weights/effnetb5/CHICAGO_1012_10:35_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model effnetb5_unet --model_path /root/weights/weights/effnetb5/ISTANBUL_1012_23:15_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model effnetb5_unet --model_path /root/weights/weights/effnetb5/MELBOURNE_1010_00:58_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel ;


python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model densenet_unet --model_path /root/weights/weights/densenet/BERLIN_1008_14:30_densenet_unet_mse_best_val_loss_2019=78.4303.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model densenet_unet --model_path /root/weights/weights/densenet/CHICAGO_1010_17:30_densenet_unet_mse_best_val_loss_2019=41.1579.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model densenet_unet --model_path /root/weights/weights/densenet/MELBOURNE_1009_16:19_densenet_unet_mse_best_val_loss_2019=25.7395.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel_and_pixel ;

#python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model densenet_unet --model_path /root/weights/weights/densenet/ISTANBUL_1011_11:39_densenet_unet_mse_best_val_loss_2019=nan_v1.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ;
