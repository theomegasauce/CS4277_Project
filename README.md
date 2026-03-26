# CS4277_Project

Current Possible Datasets:
http://deepglobe.org/
https://registry.opendata.aws/spacenet/
https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset
https://www.kaggle.com/c/cil-road-segmentation

we are on the massechusets roads dataset

here is the structure of the model so far

INPUT 512×512×3
       │
  ┌────▼────┐
  │   E1    │ → S1 (skip) ────────────────────────────────── ┐
  │ 32 filt │                                                │
  └────┬────┘                                                │
  ┌────▼────┐                                                │
  │   E2    │ → S2 (skip) ───────────────────────── ┐        │
  │ 64 filt │                                       │        │
  └────┬────┘                                       │        │
  ┌────▼────┐                                       │        │
  │   E3    │ → S3 (skip) ─────────────────┐        │        │
  │128 filt │                              │        │        │
  └────┬────┘                              │        │        │
  ┌────▼────┐                              │        │        │
  │   E4    │ → S4 (skip) ─────────┐       │        │        │
  │256 filt │                      │       │        │        │
  └────┬────┘                      │       │        │        │
  ┌────▼──────────────┐            │       │        │        │
  │       LMCM        │            │       │        │        │
  │ 1×1 | d2 | d4 | GP│            │       │        │        │
  │  → concat → fuse  │            │       │        │        │
  └────┬──────────────┘            │       │        │        │
  ┌────▼────┐                      │       │        │        │
  │   D1    │◄─────────────────────┘       │        │        │
  │128 filt │                              │        │        │
  └────┬────┘                              │        │        │
  ┌────▼────┐                              │        │        │
  │   D2    │◄─────────────────────────────┘        │        │
  │ 64 filt │                                       │        │
  └────┬────┘                                       │        │
  ┌────▼────┐                                       │        │
  │   D3    │◄──────────────────────────────────────┘        │
  │ 32 filt │                                                │
  └────┬────┘                                                │
  ┌────▼────┐                                                │
  │   D4    │◄───────────────────────────────────────────────┘
  │ 32 filt │
  └────┬────┘
       │
  ┌────┴──────────────────────┐
  │         OUTPUT HEADS       │
  ├───────────┬────────┬───────┤
  ▼           ▼        ▼
SEG HEAD   EDGE HEAD  CL HEAD
512×512×1  512×512×1  512×512×1
(road mask)(edges)   (centerline)