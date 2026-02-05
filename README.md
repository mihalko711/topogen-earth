# TopoGen Earth

TopoGen Earth is a generative topological atlas of satellite imagery,
where each region of a self-organizing map can be explored and sampled
using a variational autoencoder.

## Структура проекта

- `notebooks/` - Jupyter ноутбуки с экспериментами и прототипами
- `src/` - Исходный код проекта
  - `som_vae.py` - Реализация SOM-VAE для классификации спутниковых изображений
  - `diffusion_vae.py` - Conditional Diffusion Model для улучшения качества реконструкции
- `SOM_VAE_EuroSAT.md` - Документация по реализации SOM-VAE
- `DIFFUSION_INTEGRATION.md` - Документация по интеграции Conditional Diffusion Model

---

*Последнее обновление: 2026-02-01*
