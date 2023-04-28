MAX_FOLD=4
cd ../CTOTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task baseline --fold $fold_id --train-train-epochs 90 --train-gpus 1 5   --train-lr 0.0001 --train-workers=4
done