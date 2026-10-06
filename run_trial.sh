# Launches the federated server. The backbone is set here and only here: the
# server hands it to every client in the /join response, so the sites do not
# need matching command lines.
#
# resnet (the original model, SGD):
# python run_server.py --run-config 'backbone=resnet batch-size=4 lr=0.0001 trial-tag="20260917_test"'
#
# retfound (ViT-L, AdamW + layer decay). retfound-weights is required — the
# server's checkpoint is what seeds every site. Add retfound-finetune=false to
# freeze the encoder and train only fc_norm and the head.

python run_server.py --run-config 'backbone=retfound
  retfound-weights="/data/russ/weights/retfound_oct.safetensors"
  retfound-lr=0.001
  batch-size=4
  local-epochs=10
  save-round-weights=true
  trial-tag="20260930_retfound_lr001_weights_saved"'
