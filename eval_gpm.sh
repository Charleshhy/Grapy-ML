python ./exp/test/eval_gpm.py \
 --batch 1 \
 --gpus 1 \
 --classes 20 \
 --dataset cihp \
 --gt_path './data/datasets/CIHP_4w/Category_ids/' \
 --txt_file './data/datasets/CIHP_4w/lists/val_id.txt' \
 --resume_model './data/models/CIHP_trained.pth' \
 --output_path './result/gpm_cihp' \
 --hidden_graph_layers 256
