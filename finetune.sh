python main.py \
    --logdir experiments/objectstitch \
    --name='car_multifg' \
    --num_workers 16 \
    --devices 1 \
    --batch_size 1 \
    --num_nodes 1 \
    --base configs/murecom.yaml \
    --package_name='Cat' \
    # |& tee experiments/logs/`date +%Y%m%d%H%M%S`.log 2>&1
