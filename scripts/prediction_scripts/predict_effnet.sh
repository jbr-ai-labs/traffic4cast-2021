
python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model effnetb5_unet --model_path /root/weights/weights/effnetb5/BERLIN_1008_14:30_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.ckpt --output_folder raw_predictions --domain_adaptation none ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model effnetb5_unet --model_path /root/weights/weights/effnetb5/BERLIN_1008_14:30_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city BERLIN --model effnetb5_unet --model_path /root/weights/weights/effnetb5/BERLIN_1008_14:30_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.ckpt --output_folder raw_predictions --domain_adaptation mean_overall


python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model effnetb5_unet --model_path /root/weights/weights/effnetb5/CHICAGO_1012_10:35_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.ckpt --output_folder raw_predictions --domain_adaptation none ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model effnetb5_unet --model_path /root/weights/weights/effnetb5/CHICAGO_1012_10:35_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city CHICAGO --model effnetb5_unet --model_path /root/weights/weights/effnetb5/CHICAGO_1012_10:35_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model effnetb5_unet --model_path /root/weights/weights/effnetb5/ISTANBUL_1012_23:15_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.ckpt --output_folder raw_predictions --domain_adaptation none ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model effnetb5_unet --model_path /root/weights/weights/effnetb5/ISTANBUL_1012_23:15_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ; 

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city ISTANBUL --model effnetb5_unet --model_path /root/weights/weights/effnetb5/ISTANBUL_1012_23:15_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel ;


python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model effnetb5_unet --model_path /root/weights/weights/effnetb5/MELBOURNE_1010_00:58_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.ckpt --output_folder raw_predictions --domain_adaptation none ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model effnetb5_unet --model_path /root/weights/weights/effnetb5/MELBOURNE_1010_00:58_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.ckpt --output_folder raw_predictions --domain_adaptation mean_by_channel ;

python3 predict_to_file.py --dataset_path /root/data/traffic4cast --city MELBOURNE --model effnetb5_unet --model_path /root/weights/weights/effnetb5/MELBOURNE_1010_00:58_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.ckpt --output_folder raw_predictions --domain_adaptation mean_overall ; 





