python ./exp/refine_on_datasets_SYN/train_gpm.py \
 --batch 10 \
 --gpus 2 \
 --loadmodel ./data/models/CIHP_trained.pth \
 --lr 0.0007 \
 --classes 20 \
 --numworker 6 \
 --testInterval 10 \
 --hidden_graph_layers 256 \
 --epochs 100 \
 --dataset cihp \
 --beta_aux 1.0 \
 --beta_main 1.0