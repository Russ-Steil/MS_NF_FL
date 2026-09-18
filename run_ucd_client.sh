# on ophlappc10 — UCD client
#
# --weights-path is optional and does NOT seed training: the client is
# overwritten with the server's global weights before its first gradient step.
# When given, the local checkpoint is fingerprinted and compared against the
# server's, which catches the two sites holding different RETFound files.
FL_TOKEN=replace-me-ucd python fl_client.py \
  --site ucd \
  --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
  --cache-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls/cache/ucd_512 \
  --server http://127.0.0.1:9092 \
  --trial-tag 20260918_retfound \
  --weights-path /data/russ/weights/retfound_oct.safetensors \
  --gpu-index 0
