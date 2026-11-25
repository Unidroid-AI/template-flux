# unidroid-template-flux

Submission template for the `flux` brain (FLUX 3 Action world action model). It is not tied to a task: it serves any FLUX 3 Action package. The reference code loads the released `black-forest-labs/flux-3-action-droid` package. Copy your trained weights into `weights/` and submit.

## Quickstart

1. Put the weights in `weights/`. For the released checkpoint, follow [Fetching the weights](#fetching-the-weights). For your own, replace `weights/policy` with your package.
2. Write the hashes into `agent.yaml` (the `weights:` block starts empty):

   ```bash
   udcli submission hash --path .
   ```

3. Check the folder and run the brain locally. Run this inside the `flux3-action-v0` brain image on a Linux GPU host:

   ```bash
   brain-comm check --submission .
   mkdir -p /tmp/brain
   brain-comm serve --submission . --socket-dir /tmp/brain --replicas 1
   ```

4. Submit. See the Miner quick start in the `unidroid-cli` README (`udcli submission check`, then `udcli submission submit`). The challenge must use the `flux3-action-v0` image. Do not change the pushed weights after submitting.

## What you may modify

Everything except the file layout that `udcli submission check` enforces.

| Path | Purpose |
| --- | --- |
| `brain/policy.py` | Inference: loads the checkpoint and answers each observation. |
| `brain/adapt.py` | Observation and action conversion, joint limits. |
| `train/` | Training notes or scripts. Uploaded in the code bundle, not evaluated. |
| `tests/` | Unit tests. Run with `pytest`. |
| `agent.yaml` | Brain config. The `weights:` block is rewritten by `udcli submission hash`. |
| `weights/` | Checkpoint. Ignored by Git and pushed to your Hugging Face repo at submit time. |

## Brain specifics

`agent.yaml` sets `policy_config` to `flux3_action_droid`, the root BF16 package (`flux_subfolder` empty), and horizon 16.

The reference policy reads a left exterior, right exterior, and wrist camera, plus the arm joints and the gripper. FLUX 3 Action wants three 360x640 views, so `brain/adapt.py` resizes all three. The policy returns 32 absolute joint targets and a gripper closed fraction. The brain keeps 16 of them. The first chunk starts at row 0. Every later chunk starts at `action_offset`, because that call is answered from an observation taken half a chunk earlier. Each command is then limited to the robot's joint range and to one drive step from the measured pose, so a distant target cannot slam the arm. The joint limits and velocities in `brain/adapt.py` are for a Franka arm; replace them for another robot. Loading the checkpoint requires the flux-action runtime from the `flux3-action-v0` brain image.

Settings in `agent.yaml`:

| Key | Meaning |
| --- | --- |
| `flux_subfolder` | Package under `weights/policy`: empty (4 steps with guidance), `variants/gd` (guidance-distilled, faster), or `variants/sd` (one step). FP8r packages need an SM89 or newer GPU and are rejected. |
| `action_offset` | First action of the 32 to keep after the first chunk. Half of `horizon` matches the eval prefetch. |
| `second_exterior` | `camera` uses `cam_exterior_right`. `duplicate` copies the left exterior. `black` sends a blank frame. |
| `offload_text_encoder` | Keep the text encoder on the CPU between new instructions. Saves about 8.9 GB of GPU memory. |
| `compile` | `false` keeps inference eager. Compiled runs differ across processes by about 0.015 rad. |

### Fetching the weights

Check the model licenses and gating on Hugging Face first, and run `hf auth login` if access is gated. The brain runs offline, so the shared encoders live under `weights/` too.

```bash
hf download black-forest-labs/flux-3-action-droid --exclude 'variants/*' --local-dir weights/policy
# optional faster variant: add --include 'variants/gd/*'
python -c "import json,pathlib;d=pathlib.Path('weights/policy');p=d/'config.native.json';p=p if p.is_file() else d/'config.json';c=json.loads(p.read_text());print(c.get('video_vae_id'), c.get('text_encoder_id'))"
```

The two ids have the form `repo[:file or folder][@revision]`. Download that repo at that revision. If the config names no revision, use `main` and note the commit.

```bash
hf download black-forest-labs/flux-3-action-base --revision <revision> \
  --include 'video_vae.safetensors' --include 'text_encoder/*' --local-dir weights
```

`hf download` leaves a hidden `.cache` folder in each `--local-dir`. The check does not skip files inside it, so remove them:

```bash
find weights -name .cache -type d -prune -exec rm -rf {} +
```

`weights/policy`, `weights/text_encoder`, and `weights/video_vae.safetensors` are ignored by Git.
