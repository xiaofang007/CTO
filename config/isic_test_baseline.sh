MAX_FOLD=4
cd ../CTOTrainer

# FOLD 0-4
for fold_id in $(seq 0 1 $MAX_FOLD)
do  
python test.py --task baseline --fold $fold_id  --test-gpus 0
done