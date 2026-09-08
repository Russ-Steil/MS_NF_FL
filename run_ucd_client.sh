# on ophlappc10 — UCD client
FL_TOKEN=replace-me-ucd python fl_client.py \
  --site ucd \
  --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
  --cache-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls/cache/ucd_512 \
  --server http://127.0.0.1:9092 \
  --trial-tag 090726_test \
  --gpu-index 0