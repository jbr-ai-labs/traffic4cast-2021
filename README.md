# 3rd Place Solution of Traffic4Cast 2021 Core Challenge

This is the code for our solution to the NeurIPS 2021 Traffic4Cast Core Challenge.

Our solution is described in the "Solving Traffic4Cast Competition with U-Net and Temporal Domain Adaptation" [paper](https://github.com/jbr-ai-labs/traffic4cast-2021/blob/dev/resources/technical_report.pdf).

### Learnt parameters
The models' learnt parameters are available by the link: https://drive.google.com/file/d/1zD0CecX4P3v5ugxaHO2CQW9oX7_D4BCa/view?usp=sharing    
Please download the archive and unzip it into the ```weights``` folder of the repository, so its structure looks like the following:

    ├── ...
    ├── traffic4cast
    ├── weights
    │   ├── densenet                 
    │   │   ├── BERLIN_1008_1430_densenet_unet_mse_best_val_loss_2019=78.4303.pth                     
    │   │   ├── CHICAGO_1010_1730_densenet_unet_mse_best_val_loss_2019=41.1579.pth
    │   │   └── MELBOURNE_1009_1619_densenet_unet_mse_best_val_loss_2019=25.7395.pth    
    │   ├── effnetb5
    │   │   ├── BERLIN_1008_1430_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.pth    
    │   │   ├── CHICAGO_1012_1035_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.pth
    │   │   ├── ISTANBUL_1012_2315_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.pth    
    │   │   └── MELBOURNE_1010_0058_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.pth    
    │   └── unet
    │       ├── BERLIN_0806_1425_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.pth    
    │       ├── CHICAGO_0805_0038_vanilla_unet_mse_best_val_loss_2019=42.6634.pth
    │       ├── ISTANBUL_0805_2317_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.pth
    │       └── MELBOURNE_0804_1942_vanilla_unet_mse_best_val_loss_2019=26.7588.pth
    ├── ...

### Submission reproduction
To generate the submission file, please run the following script:
```shell
# $1 - absolute path to the dataset, $2 device to run inference
bash submission.sh {absolute path to dataset} {cpu, cuda}
# Launch example
bash submission.sh /root/data/traffic4cast cuda
```
The above sctipt generates the submission file ```submission/submission_all_unets_da_none_mpcpm1_mean_temporal_{date}.zip```, which gave us the best MSE of ```49.379068541527``` on the final leaderboard.
