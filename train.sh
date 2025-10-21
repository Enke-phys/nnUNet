#Preprocess data
nnUNetv2_plan_and_preprocess -d 1  --verify_dataset_integrity

#Train model

nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz

 #2D UNet
 nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]

 #3D full resolution U-Net
 nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]

 #3D low resolution U-Net
 nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD [--npz]

 #3D full resolution U-Net
 nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD [--npz]

#Usinf multiple GPUs for training
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 [--npz] & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 [--npz] & # train on GPU 2
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 [--npz] & # train on GPU 3
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 [--npz] & # train on GPU 4
