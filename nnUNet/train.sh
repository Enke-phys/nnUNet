#Preprocess data
nnUNetv2_plan_and_preprocess -d 1  --verify_dataset_integrity

#Train model
#nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz

 #2D UNet
 #nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD --npz

 #3D full resolution U-Net
 #nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD --npz

 #3D low resolution U-Net
 #nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD --npz

 #3D full resolution U-Net
 #nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD --npz

#Usinf multiple GPUs for training
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz & # train on GPU 0
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 --npz & # train on GPU 1
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 --npz & # train on GPU 2
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 --npz & # train on GPU 3
#CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 --npz & # train on GPU 4

#Find best configuration
#   nnUNetv2_find_best_configuration 1 -c 3d_fullres 

# Dice berechnung
#   nnUNetv2_predict \
-i $nnUNet_raw/$dataset_name/imagesTs \
-o $nnUNet_results/$dataset_name/nnUNetTrainer__nnUNetPlans__3d_fullres/predicted_test_images \
-d $dataset_name \
-f 0 \
-c 3d_fullres

#Inferenz
#   nnUNetv2_predict \
#  -i nnUNetDataset/nnUNet_raw/Dataset001_MRT/imagesTs \
#  -o nnUNetDataset/predicted_test_images \
#  -d Dataset001_MRT \
#  -f 0 \
#  -c 3d_fullres


#Postprocessing
#   nnUNetv2_apply_postprocessing \
#  -i nnUNetDataset/nnUNet_results/nnUNetTrainer__nnUNetPlans__3d_fullres/Dataset001_MRT/fold_0/validation_raw \
#  -o nnUNetDataset/nnUNet_results/nnUNetTrainer__nnUNetPlans__3d_fullres/Dataset001_MRT/fold_0/validation_postprocessed \
#  --pp_pkl_file nnUNetDataset/nnUNet_results/nnUNetTrainer__nnUNetPlans__3d_fullres/Dataset001_MRT/postprocessing.pkl \
#  -plans_json nnUNetDataset/nnUNet_results/nnUNetTrainer__nnUNetPlans__3d_fullres/Dataset001_MRT/plans_3D_fullres_plans_...json \
#  -dataset_json nnUNetDataset/nnUNet_results/nnUNetTrainer__nnUNetPlans__3d_fullres/Dataset001_MRT/dataset.json


