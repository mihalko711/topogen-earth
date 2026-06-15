# TopoGen Earth

**Generative topological atlas of satellite imagery**

TopoGen Earth — проект по генерации спутниковых изображений с использованием **Flow Matching** diffusion models на базе [HuggingFace Diffusers](https://github.com/huggingface/diffusers).

---

## Focus

- **Flow Matching** for satellite image generation (EuroSAT)
- Pre-trained and custom pipelines via `diffusers`
- Conditional generation (class-conditional, text-conditional)
- Exploration of Riemannian Flow Matching on geographic manifolds

## Dataset

**EuroSAT** — 21 000 satellite images, 10 land-cover classes, 64×64 RGB.

## Structure

```
topogen-earth/
├── src/          # Source code
├── notebooks/    # Jupyter notebooks
└── README.md
```

## Quick Start

```bash
pip install diffusers torch transformers accelerate datasets
```

See notebooks for usage examples.

---

*Last updated: 2026-06-15*
