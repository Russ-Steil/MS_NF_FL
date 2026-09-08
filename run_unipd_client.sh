# on Mattia's SLURM node — UniPD client
FL_TOKEN=replace-me-unipd python fl_client.py \
  --site unipd \
  --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
  --cache-path /data/giacomo/Hereditary_MS/noflower_FL/test_files/unipd_fake_cache \
  --server http://140.226.4.75:9092 \
  --trial-tag 090726_test \
  --gpu-index 0