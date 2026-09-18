# on Mattia's SLURM node — UniPD client
#
# --weights-path points at UniPD's own copy of the RETFound checkpoint; it does
# not have to sit where the server's does. It is optional and does NOT seed
# training — the client is overwritten with the server's global weights before
# its first gradient step. When given, the local checkpoint is fingerprinted and
# compared against the server's, which catches the two sites holding different
# RETFound files.
#
# Request at least 24 GB of host RAM for a retfound run: each weight exchange
# transiently holds ~3.6 GB outside the GPU.
FL_TOKEN=replace-me-unipd python fl_client.py \
  --site unipd \
  --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
  --cache-path /data/giacomo/Hereditary_MS/noflower_FL/test_files/unipd_fake_cache \
  --server http://140.226.4.75:9092 \
  --trial-tag 20260918_retfound \
  --weights-path /path/on/unipd/retfound_oct.safetensors \
  --gpu-index 0
