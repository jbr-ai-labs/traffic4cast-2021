#!/bin/bash

# Script for generating the submission file
echo "    Reading data from: " "$1"
echo "    Running on device: " "$2"

# No Domain Adaptation

# BERLIN
python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model densenet_unet \
  --model_path weights/densenet/BERLIN_1008_1430_densenet_unet_mse_best_val_loss_2019=78.4303.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/BERLIN_0806_1425_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/BERLIN_1008_1430_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

# CHICAGO
python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model densenet_unet \
  --model_path weights/densenet/CHICAGO_1010_1730_densenet_unet_mse_best_val_loss_2019=41.1579.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/CHICAGO_0805_0038_vanilla_unet_mse_best_val_loss_2019=42.6634.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/CHICAGO_1012_1035_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

# ISTANBUL
python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/ISTANBUL_0805_2317_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/ISTANBUL_1012_2315_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

# MELBOURNE

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model densenet_unet \
  --model_path weights/densenet/MELBOURNE_1009_1619_densenet_unet_mse_best_val_loss_2019=25.7395.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/MELBOURNE_0804_1942_vanilla_unet_mse_best_val_loss_2019=26.7588.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/MELBOURNE_1010_0058_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.pth \
  --output_folder predictions \
  --domain_adaptation none \
  --device "$2" ;

# With Temporal Domain Adaptation
# BERLIN
python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model densenet_unet \
  --model_path weights/densenet/BERLIN_1008_1430_densenet_unet_mse_best_val_loss_2019=78.4303.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/BERLIN_0806_1425_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/BERLIN_1008_1430_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

# CHICAGO
python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model densenet_unet \
  --model_path weights/densenet/CHICAGO_1010_1730_densenet_unet_mse_best_val_loss_2019=41.1579.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/CHICAGO_0805_0038_vanilla_unet_mse_best_val_loss_2019=42.6634.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/CHICAGO_1012_1035_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

# ISTANBUL
python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/ISTANBUL_0805_2317_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/ISTANBUL_1012_2315_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

# MELBOURNE

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model densenet_unet \
  --model_path weights/densenet/MELBOURNE_1009_1619_densenet_unet_mse_best_val_loss_2019=25.7395.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model unet \
  --model_path weights/unet/MELBOURNE_0804_1942_vanilla_unet_mse_best_val_loss_2019=26.7588.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;

python3 predict_to_file.py \
  --dataset_path "$1" \
  --city BERLIN \
  --model effnetb5_unet \
  --model_path weights/effnetb5/MELBOURNE_1010_0058_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.pth \
  --output_folder predictions \
  --domain_adaptation mean_by_channel_and_pixel \
  --device "$2" ;


# Generate the submission file based on ensembling the predictions from above:
python3 scripts/generate_submission_from_files.py --dataset_path "$1"
