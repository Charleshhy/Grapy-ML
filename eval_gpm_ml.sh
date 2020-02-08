python ./exp/test/eval_gpm_ml.py \
 --batch 1 \
 --gpus 1 \
 --classes 7 \
 --dataset pascal \
 --gt_path './data/datasets/pascal/SegmentationPart/' \
 --txt_file './data/datasets/pascal/list/val_id.txt' \
 --resume_model './data/models/GPM-ML_finetune_PASCAL.pth' \
 --output_path './result/gpm_ml_pascal' \
 --hidden_graph_layers 256
