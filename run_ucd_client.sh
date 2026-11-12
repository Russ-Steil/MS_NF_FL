# on ophlappc10 — UCD client
#
# --weights-path is optional and does NOT seed training: the client is
# overwritten with the server's global weights before its first gradient step.
# When given, the local checkpoint is fingerprinted and compared against the
# server's, which catches the two sites holding different RETFound files.
#
# The final global weights are written to runs/fl_clients/ucd/<trial>/ when the
# run finishes; pass --save-path to put them somewhere else.
FL_TOKEN=158a6bef3ec90e23620d9d5bb3fd0a3f python fl_client.py \
  --site ucd \
  --data-path /data/russ/MS_NF_FL/symlinks/nonretina_ms_matched \
  --cache-path /data/russ/MS_NF_FL/ucd_512_nonretina_matched_cache \
  --server https://127.0.0.1:9092 \
  --ca-cert certs/ca.crt \
  --trial-tag nonretina_matched_10122026 \
  --weights-path /data/russ/weights/retfound_oct.safetensors \
  --gpu-index 0
