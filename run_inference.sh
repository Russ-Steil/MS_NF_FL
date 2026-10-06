# on ophlappc10 — score a federated checkpoint on the held-out test split.
#
# --backbone must name the architecture the checkpoint was trained with. The
# server writes a bare state_dict with nothing in it to say which that is, so a
# retfound checkpoint has to be labelled here or it is assumed to be a resnet
# and fails on key mismatch. (The per-site copies fl_client.py saves do record
# their backbone and input size, and load without the flag.)
#
# --image-size stays 512 for both: it is the decode/cache size, not the model
# input. RETFound's second resize to 224x448 is applied on top of it.
#
# --labels-csv supplies the age column for the age vs p(MS) correlation. It
# defaults to <data-path>/labels.csv, so it is only worth naming when the ages
# live elsewhere; pass 'none' to skip the age analysis.
#
# --val-split is the split the entropy uncertainty analysis (entropy_uq.py,
# written to <out-dir>/youden and youden_patient) picks its Youden threshold on
# before applying it to --split. Defaults to val; pass 'none' to skip it.

# retfound (ViT-L/16 at 224x448)
python inference.py \
  --checkpoint /data/russ/MS_NF_FL/results/20260918_retfound_lr0001/model_best_val_auc.pth \
  --backbone retfound \
  --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
  --split test \
  --val-split val \
  --site ucd \
  --batch-size 16 \
  --labels-csv /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls/labels.csv \
  --gpu-index 0

# resnet50
# python inference.py \
#   --checkpoint results/090726_test/model_best_val_auc.pth \
#   --backbone resnet \
#   --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
#   --split test \
#   --site ucd \
#   --batch-size 16 \
#   --gpu-index 0
