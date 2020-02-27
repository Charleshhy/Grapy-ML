python ./exp/single_syn/train_dlab.py \
 --batch 10 \
 --gpus 2 \
 --loadmodel ./data/models/deeplab_v3plus_v3.pth \
 --lr 0.007 \
 --classes 20 \
 --numworker 6 \
 --testInterval 10 \
 --epochs 100 \
 --dataset cihp