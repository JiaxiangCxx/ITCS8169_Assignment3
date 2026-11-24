# One-shot scene classification with CLIP and SigLIP

ITCS 6169/8169 Computer Vision, Assignment 3. The task is to classify 16 scene categories using **exactly one labeled training image per class**.

**Test accuracy: 94.00%** (400 test images, main one-shot split seed 0).
Over 10 random one-shot splits the same method gives **94.50 ± 0.56%**.

| Setting (CLIP ViT-L/14-336 + SigLIP2 SO400M/512) | Test accuracy |
|---|---|
| Zero-shot (text prompts only, no training image) | 89.75% |
| One-shot, text + image prototype (inductive) | 90.40 ± 1.62% |
| **One-shot + test-time soft k-means (final)** | **94.50 ± 0.56%** (seed 0: **94.00%**) |

## Method

The image and text encoders stay frozen, so nothing is fine-tuned. All steps work on L2-normalised embeddings.

1. **Text classifier.** Every class name gets 1–3 synonyms (for example, *OpenCountry* becomes "open country", "an open countryside field", or "a rural landscape"). Each synonym fills 8 templates, 3 of which mention black-and-white or grayscale photos because the dataset is grayscale. The prompt embeddings are averaged into one vector per class.
2. **One-shot prototype.** The single training image of each class is encoded as 34 views: center crop, horizontal flip, and 32 random crops with color jitter. Their mean is the class prototype.
3. **Fusion.** Image-to-text similarities are much smaller than image-to-image similarities (the CLIP modality gap). The two score matrices are therefore standardized per image (z-score over the 16 classes) and then added.
4. **Test-time adaptation (transductive).** The unlabeled test images are used to refine one image-space centroid per class, by soft k-means for 20 iterations:
   - soft assignments `q = softmax(κ · (z(text scores) + z(image-centroid scores)))`;
   - centroid update `μ_k = norm(norm(Σ_i q_ik x_i) + prototype_k)`.

   The text scores act as a fixed prior, and each centroid stays anchored to its one training image. No test labels are used.
5. **Backbone ensemble.** The normalised embeddings of CLIP ViT-L/14-336 (OpenAI) and SigLIP2 SO400M/16-512 (WebLI) are concatenated, each scaled by 1/√2.

Hyperparameters were set before any evaluation and were not tuned on test labels: κ = 2, image weight 1, 20 iterations. See `src/methods.py`.

## The one-image-per-class rule

- `make_splits.py` picks one training image per class with a fixed RNG seed. `splits/oneshot_seed0.json` is the split used for the reported model.
- Feature extraction reads **only** the training images listed in `splits/*.json`; no other training image is ever opened.
- Test-time adaptation uses the unlabeled test images, never their labels.
- Seeds 1–9 exist only to measure how much the choice of the single image matters.

Seed-0 training images: Bedroom/image_0177, Coast/image_0199, Flower/image_0127, Forest/image_0041, Highway/image_0047, Industrial/image_0007, InsideCity/image_0012, Kitchen/image_0003, LivingRoom/image_0027, Mountain/image_0285, Office/image_0105, OpenCountry/image_0342, Store/image_0094, Street/image_0114, Suburb/image_0232, TallBuilding/image_0234.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/download_data.py --dest data      # course Google Drive folder -> data/{train,test,test2}
```

**A. Inference with the saved model** (one GPU, about 2 minutes; the backbones download automatically):

```bash
python predict.py --model checkpoints/oneshot_clipL_siglip2.npz --images data/test  --out predictions/test_predictions.csv
# -> accuracy: 94.00%
python predict.py --model checkpoints/oneshot_clipL_siglip2.npz --images data/test2 --out predictions/test2_predictions.csv
```

`checkpoints/oneshot_clipL_siglip2.npz` is the trained model: for each backbone, the 16 text-class embeddings and the embeddings of the 16 one-shot images (34 views each). `predict.py` encodes a folder of images, runs the test-time adaptation on that folder, and writes a CSV. It also prints the accuracy when the folder has one sub-folder per class.

**B. The full results table from cached features** (CPU only, seconds):

```bash
python evaluate.py --models clip_l14_336 siglip2_so400m_512 clip_l14_336+siglip2_so400m_512
```

**C. Everything from scratch** (8 GPUs, about 30 minutes):

```bash
python make_splits.py                 # splits/oneshot_seed{0..9}.json
bash scripts/extract_all.sh           # cache embeddings of 8 backbones, one GPU each
python evaluate.py --models clip_l14_336 dfn5b_h14_378 siglip_so400m_384 siglip2_so400m_512 \
    laion_bigg14 eva02_e14_plus pe_core_bigg_448 siglip2_gopt_384 clip_l14_336+siglip2_so400m_512
python build_model.py --models clip_l14_336+siglip2_so400m_512 --method softkmeans --seed 0 \
    --out checkpoints/oneshot_clipL_siglip2.npz
```

## All backbones

Test accuracy (%). Each cell is the mean ± std over 10 one-shot splits; seed 0 is in brackets.

| Backbone | Zero-shot | Prototype only | Text + proto | Tip-Adapter | **Test-time soft k-means** |
|---|---|---|---|---|---|
| CLIP ViT-L/14-336 (OpenAI) | 90.50 | 72.15 ± 6.83 | 89.60 ± 1.51 | 90.60 ± 0.17 | 93.70 ± 0.56 [93.25] |
| SigLIP2 SO400M/16-512 | 89.00 | 75.30 ± 5.37 | 89.55 ± 2.10 | 89.38 ± 0.28 | 93.60 ± 1.03 [92.75] |
| SigLIP2 g-opt/16-384 | 89.75 | 74.83 ± 3.69 | 89.53 ± 2.06 | 89.80 ± 0.38 | 93.42 ± 1.11 [92.25] |
| DFN5B ViT-H/14-378 | 87.75 | 79.78 ± 4.48 | 90.12 ± 2.18 | 87.85 ± 0.17 | 93.10 ± 1.09 [92.50] |
| SigLIP SO400M/14-384 | 87.75 | 76.40 ± 4.65 | 88.20 ± 2.14 | 87.78 ± 0.17 | 92.60 ± 1.40 [93.00] |
| EVA02-E/14+ (LAION-2B) | 88.00 | 77.30 ± 4.59 | 89.47 ± 2.63 | 88.15 ± 0.12 | 92.45 ± 2.74 [92.50] |
| OpenCLIP ViT-bigG/14 (LAION-2B) | 87.75 | 77.53 ± 4.82 | 89.08 ± 2.07 | 87.75 ± 0.00 | 92.28 ± 1.95 [92.75] |
| PE-Core bigG/14-448 (Meta) | 87.25 | 73.22 ± 4.82 | 88.85 ± 2.21 | 87.67 ± 0.23 | 91.47 ± 1.73 [92.25] |
| **CLIP-L + SigLIP2-SO400M** | 89.75 | 75.95 ± 6.17 | 90.40 ± 1.62 | 90.05 ± 0.15 | **94.50 ± 0.56 [94.00]** |
| All 8 backbones | 88.00 | 78.75 ± 4.47 | 90.08 ± 2.12 | 88.20 ± 0.22 | 93.65 ± 1.22 [93.50] |

The raw numbers, including the sensitivity runs, are in `results/*.jsonl`.

## AI usage

AI was used for polishing and idea gathering.

## Layout

```
src/common.py         classes, prompts, backbone list, one-shot split sampling
src/methods.py        zero-shot, prototype, fusion, Tip-Adapter, soft k-means, label propagation
extract_features.py   cache image/text embeddings for one backbone
evaluate.py           accuracy of every method over the 10 one-shot splits
build_model.py        save the final one-shot model (checkpoints/)
predict.py            inference from raw images with the saved model
splits/               the one-shot training images per seed
features/             cached embeddings of the two final backbones
predictions/          CSV predictions for test and test2
```
