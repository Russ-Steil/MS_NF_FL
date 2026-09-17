# on ophlappc10 — score the best federated checkpoint on the held-out test split
python inference.py \
  --checkpoint results/090726_test/model_best_val_auc.pth \
  --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
  --split test \
  --site ucd \
  --batch-size 16 \
  --gpu-index 0
