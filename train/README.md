# Fine-tuning

Evaluation does not run this directory.

`weights/policy` is the released `black-forest-labs/flux-3-action-droid` package. `weights/video_vae.safetensors` and `weights/text_encoder` are the shared frozen encoders from `black-forest-labs/flux-3-action-base`. `agent.yaml` lists each file's sha256. `policy_config` is `flux3_action_droid`.

To fine-tune further, follow `docs/droid-finetune.md` in the flux-action repository, with data from `task-franka-tabletop/scripts/record_demos.py` indexed as described there. Export the result with `flux-action export-checkpoint --dtype bfloat16`, place it under `weights/policy`, and update the hashes in `agent.yaml`. Keep the `droid` camera layout, 8-D absolute joint actions, and 32-step chunk.
