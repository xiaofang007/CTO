GPU=1 6
MAX_FOLD=4
cd ../CTOTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python train.py --task baseline --fold $fold_id --train-gpus $GPU --train-lr 0.0001 --train-train-epochs 100
done