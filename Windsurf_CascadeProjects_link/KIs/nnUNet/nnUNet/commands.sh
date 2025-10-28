#Start training
echo "Starte Training (DDP, 5 GPUs)..."
torchrun --nproc_per_node=5 train.py --dataroot ./datasets/t1t2 --name t1t2_cyclegan --model cycle_gan --direction AtoB --input_nc 1 --output_nc 1 --batch_size 1 --num_threads 2 --n_epochs 100 --n_epochs_decay 100 --save_epoch_freq 5 --no_dropout --no_html --norm instance --preprocess resize_and_crop --load_size 320 --crop_size 320
echo "Training abgeschlossen!"

# Inferenz starten
# echo "Starte Inferenz A->B (T2->T1)"
# python3 test.py --dataroot ./datasets/t1t2 --name t1t2_cyclegan --model cycle_gan --direction AtoB --input_nc 1 --output_nc 1 --num_threads 0 --batch_size 1 --no_dropout --preprocess resize_and_crop --load_size 320 --crop_size 320
# echo "Starte Inferenz B->A (T1->T2)"
# python3 test.py --dataroot ./datasets/t1t2 --name t1t2_cyclegan --model cycle_gan --direction BtoA --input_nc 1 --output_nc 1 --num_threads 0 --batch_size 1 --no_dropout --preprocess resize_and_crop --load_size 320 --crop_size 320
# echo "Inferenzen abgeschlossen"