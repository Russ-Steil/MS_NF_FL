# UCD, once
# python cache_images.py \
#   --site ucd \
#   --data-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls \
#   --cache-path /data/giacomo/Hereditary_MS/data/MS_cohort_non_MS_ctrls/cache/ucd_512 \
#   --image-size 512 \
#   --workers 16

python cache_images.py \
  --site ucd \
  --data-path /data/russ/MS_NF_FL/symlinks/nonretina_ms_matched \
  --cache-path /data/russ/MS_NF_FL/ucd_512_nonretina_matched_cache \
  --image-size 512 \
  --workers 16