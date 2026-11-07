# SubCulture-2.6K

Code for *Beyond a Single Score: An Audit of Aesthetic Evaluation in Text-to-Image Pipelines Across Subcultural Visual Languages* (NeurIPS 2026, Evaluations and Datasets Track).

Text-to-image models are used by online creative communities to generate imagery in shared visual languages, from Cottagecore's pastoral warmth to Cyberpunk's neon dystopias. The aesthetic predictors embedded in those pipelines filter training data and score generated outputs. SubCulture-2.6K tests whether the predictors treat six of these visual languages alike: Cottagecore, Cyberpunk, Dark Academia, Goblincore, Synthwave, and Y2K. The dataset holds 2,610 Stable Diffusion image–prompt pairs, image features, and per-image scores from four deployed predictors (LAION-Aesthetics V2, PickScore, ImageReward, and HPS-v2).

This repository contains the code. The dataset is on the Hugging Face Hub at [DanLi26/subculture-2.6k](https://huggingface.co/datasets/DanLi26/subculture-2.6k).

## Requirements

Python 3.10 or later. The analyses were run with Python 3.13 on macOS and need no GPU.

```bash
pip install -r requirements.txt
```

PickScore, ImageReward and HPS-v2 need conflicting versions of `transformers` and `timm`. Only rescoring needs them; each has a pinned environment in `environments/`:

```bash
python -m venv .venv-pick && .venv-pick/bin/pip install -r environments/subc-pick.requirements.txt
python -m venv .venv-ir   && .venv-ir/bin/pip   install -r environments/subc-ir.requirements.txt
python -m venv .venv-hps  && .venv-hps/bin/pip  install -r environments/subc-hps.requirements.txt
```

## Data

Download the dataset and point `SUBCULTURE_DATA` at it:

```bash
huggingface-cli download DanLi26/subculture-2.6k --repo-type dataset --local-dir subculture-2.6k
export SUBCULTURE_DATA=$PWD/subculture-2.6k
python3 verify/check_release.py
```

If `SUBCULTURE_DATA` is unset, the scripts look for a folder named `dataset/` beside this repository. Scripts only read from the dataset. Everything they write goes to `output/` in this repository.

## Pretrained models

The analyses in the table below start from the released scores and embeddings and need no model weights. Rescoring the images needs:

| Predictor or encoder | Checkpoint |
|---|---|
| CLIP ViT-L/14 (embeddings, LAION-A input) | `open_clip` `ViT-L-14`, `pretrained="openai"` |
| LAION-Aesthetics V2 | [`sac+logos+ava1-l14-linearMSE.pth`](https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth), saved to `models/` |
| PickScore | [`yuvalkirstain/PickScore_v1`](https://huggingface.co/yuvalkirstain/PickScore_v1), processor `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| ImageReward | `ImageReward-v1.0` via the `image-reward` package |
| HPS-v2 | [`xswu/HPSv2`](https://huggingface.co/xswu/HPSv2), `hps_version="v2.1"` via the `hpsv2` package |

Hugging Face checkpoints download on first use. The LAION-A head must be downloaded by hand:

```bash
mkdir -p models
curl -L -o models/sac+logos+ava1-l14-linearMSE.pth \
  https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth
```

## Reproducing the paper

Each command below reads the released dataset and writes to `output/`. On a laptop CPU the whole list finishes in a few minutes.

| Paper result | Command | Main output |
|---|---|---|
| Section 5.1, color-only separability | `python3 results/separability/cross_subculture_color_analysis.py` | `output/color_classification_report.txt` |
| Section 5.1, CLIP separability and confusions | `python3 results/separability/clip_subculture_analysis.py` | `output/clip_classification_report.txt`, `output/feature_comparison_table.csv` |
| Section 5.2, subculture means, $\eta^2$, cutoff retention | `python3 results/laion_profile/cross_scorer_analysis.py` | `output/rq2b_stats_report.txt` |
| Section 5.2, Table 4, residual $\eta^2$ | `python3 results/laion_profile/rq2c_regression.py` | `output/rq2c_linear_coefs.csv`, `output/rq2c_residual_anova.csv` |
| Section 5.3, matched text inputs | `python3 results/matched_text/analyze_matched_text.py --arm {empty,const,label}` | `output/matched_<arm>_report.txt` |
| Table 5 and Appendix D, bootstrap intervals | `python3 results/matched_text/bootstrap_agreement.py` | `output/bootstrap_report.md` |
| Peer $R^2$ and within-subculture $\rho$ under each text input | `python3 results/matched_text/matched_arm_axes.py` | `output/matched_arm_report.md` |
| Table 3 and Figure 1 | `python3 figures/rebuild_displays.py` | `output/table3_an_image.csv`, `output/fig1_an_image.png` |
| Section 5.4 and Table 6, photographs and other generators | `python3 controls/summarize_controls.py` | `output/controls_summary.txt`, `output/table6_generators.csv` |
| Appendix G, near-duplicates | `python3 audit/near_duplicates/dedup_analysis.py`, then `check_exact_pair_bytes.py` | `output/dedup_report.txt` |
| Appendix G, label check | `python3 audit/label_check/analyze_label_audit.py` | `output/label_check_report.txt` |
| Appendix H, scrape order | `python3 audit/scrape_order/corpus_independence.py` | `output/order_report.txt` |

The text input `const` is the fixed text "an image"; the released files and tables call it `an_image`.

Some scripts read tables that an earlier one writes, so run them in this order:

```bash
python3 results/laion_profile/cross_scorer_analysis.py
python3 results/laion_profile/rq2c_regression.py
python3 results/separability/cross_subculture_color_analysis.py
python3 results/separability/clip_subculture_analysis.py
for arm in empty const label; do python3 results/matched_text/analyze_matched_text.py --arm $arm; done
python3 results/matched_text/bootstrap_agreement.py
python3 results/matched_text/matched_arm_axes.py
python3 figures/rebuild_displays.py
python3 controls/summarize_controls.py
python3 audit/near_duplicates/dedup_analysis.py
python3 audit/near_duplicates/check_exact_pair_bytes.py
python3 audit/label_check/analyze_label_audit.py
python3 audit/scrape_order/corpus_independence.py
```

`bootstrap_agreement.py`, `matched_arm_axes.py` and `rebuild_displays.py` will not overwrite their earlier outputs, so clear `output/` before running them again. The bootstrap uses 2,000 replicates with seed 20261021 and reproduces the released tables in the dataset's `bootstrap/` folder exactly.

## Rebuilding the released data

These scripts rebuild the released files from the images. They need the model weights above, and their outputs go to `output/`, not the dataset.

| Released file | Command |
|---|---|
| `data/clip_embeddings.npy`, `data/clip_metadata.csv` | `python3 dataset/extract_clip_embeddings.py` |
| `data/color_analysis/` (K-means, k = 5, in RGB) | `python3 dataset/analyze_colors.py` |
| `data/interpretable_features.csv` | `python3 dataset/extract_interpretable_features.py` |
| `data/scores/` | `bash scoring/run_all_scorers.sh laion`, then `pick`, `ir` and `hps`, each in its own environment |
| `matched_text/` | `bash results/matched_text/run_matched_text.sh {empty,const,label} {pick,ir,hps}` |
| `cross_generator/scores.csv` | `python3 controls/generators/score_generators.py` |
| Separability in the PickScore and ImageReward encoders (Section 5.1) | `python3 results/separability/extract_backbone_embeddings.py --backbone {vith_pick,blip_ir}`, then `classify_backbones.py` |

To analyse your own rescored files, pass `--scores-dir output/matched_text` to `analyze_matched_text.py`.

## Inputs that are not released

- **Photographs.** Real photographs are released as URLs, licenses and scores only (`photographs/photographs.csv`). To rescore them, download the files into `controls/photographs/images/{y2k,cyberpunk}/` and run `python3 controls/photographs/score_reference_set.py --stage score`. `fetch_candidates_openverse.py`, `dedup_candidates.py` and `apply_decisions.py` are the collection and filtering steps described in Appendix E.
- **Generator API calls.** `controls/generators/generate_images.py` needs OpenAI, Google and Ideogram API keys (read from `controls/generators/.env`). The 182 generated images are released in `cross_generator/images/`.
- **Prompt matching.** `dataset/parse_prompts.py` reads the per-subculture prompt text files from the original collection, which are not released. Its result is `data/prompts/prompts_with_flags.csv`, flagged by `dataset/scan_prompts_nsfw.py`.
- **Label-check sheets.** `audit/label_check/make_*.py` build the coding sheets given to the two coders; the coded results are in `label_check/`.
- The Lexica collection script is not released.

## License

The code is released under the [MIT license](LICENSE). The dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The 182 cross-generator images are outputs of commercial APIs and are not covered by the dataset license; the photographs remain under the licenses listed with each URL.

## Citation

```bibtex
@inproceedings{li2026beyond,
  title     = {Beyond a Single Score: An Audit of Aesthetic Evaluation in Text-to-Image Pipelines Across Subcultural Visual Languages},
  author    = {Li, Dan and Fan, Ling and Xia, Lei and Ahmed, Mudasir and Pei, Yuying},
  booktitle = {Advances in Neural Information Processing Systems, Evaluations and Datasets Track},
  year      = {2026}
}
```
