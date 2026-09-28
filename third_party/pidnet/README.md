# Vendored PIDNet model code

Source: <https://github.com/XuJiacong/PIDNet> (MIT, see `LICENSE`), commit
`4c158cf24ce432f0a8cb43364fae38d93cee0dc3` (2025-12-18).

| File | Upstream path | Modifications |
|---|---|---|
| `pidnet.py` | `models/pidnet.py` | none (byte-identical) |
| `model_utils.py` | `models/model_utils.py` | none (byte-identical) |
| `__init__.py` | — | added here; re-exports `PIDNet`, `get_pred_model` |
| `LICENSE` | `LICENSE` | none |

Only the network definition is vendored. The upstream training framework (configs, datasets,
criterion, `FullModel`) is **not** copied; the loss, edge targets and training loop used in the
thesis are re-implemented in `src/firecls/baselines/pidnet/` following upstream semantics
(`utils/criterion.py`, `utils/utils.py::FullModel`, `datasets/base_dataset.py::gen_sample`).

PIDNet-S is instantiated with the upstream `get_seg_model` arguments for the `s` variant:
`PIDNet(m=2, n=3, planes=32, ppm_planes=96, head_planes=128)`.

## ImageNet-pretrained weights

The upstream README links the ImageNet-pretrained PIDNet-S backbone
(`PIDNet_S_ImageNet.pth.tar`):

* file: <https://drive.google.com/file/d/1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-/view?usp=sharing>
* the upstream README warns that individual links may have stopped working and points to the
  shared folder instead:
  <https://drive.google.com/drive/folders/0BySIOtxxULinfjlGdGFiT3NQVUdLVDBxWnhhTjB4VXNBRkFOa281WHlkektYY2VBcWVZb1k?resourcekey=0-w0JIXUekD-FCW-Rm1Z-HfQ&usp=sharing>

Download it manually to `third_party/weights/PIDNet_S_ImageNet.pth.tar` (git-ignored) and pass
`--pretrained third_party/weights/PIDNet_S_ImageNet.pth.tar` to `scripts/pidnet/train_pidnet.py`.
Loading follows upstream `get_seg_model(..., imgnet_pretrained=True)`: tensors whose name and
shape match are copied, the rest (segmentation heads) keep their random initialisation. The
number of copied tensors and the file's SHA-256 are stored in the training checkpoint.
