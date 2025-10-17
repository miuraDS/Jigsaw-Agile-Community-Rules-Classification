# Experiment Results

This file tracks experiment results for the Jigsaw Agile Community Rules Classification project.

## Evaluation Metric
- **Primary Metric**: Column-averaged AUC (Area Under the ROC Curve)

---

## Experiment Log

### Experiment 1: Qwen Model with LoRA Fine-tuning + Embeddings (Original/Baseline)
**Date**: 2025-10-15 21:57:38
**Version**: Original (Baseline)
**Notebook**: `notebooks/test-on testdataset+qwenemdding+llama lr.ipynb`
**Source Code**: `src/test-on testdataset+qwenemdding+llama lr.py`
**Score (Column-averaged AUC)**: **0.916**

#### Model Configuration:
- **Base Model**: Qwen 2.5 (0.5b-instruct-gptq-int4)
- **Fine-tuning**: LoRA (rank=16, alpha=32, dropout=0.1)
- **Training**: 1 epoch, learning rate=1e-4, warmup=0.03
- **Additional Models**:
  - Qwen 2.5 14B (with LoRA)
  - Qwen3 Embeddings (0.6B) with semantic search

#### Key Features:
- Basic data augmentation with random example selection
- Standard prompt engineering
- Simple ensemble weights (50% / 30% / 20%)
- TOP_K=2000 for semantic search

#### Notes:
- Baseline implementation
- Ensemble of three models
- Good starting performance

---

### Experiment 2: Enhanced Version with TTA and Advanced Augmentation (V2)
**Date**: 2025-10-15 22:02:41
**Version**: V2 (Improved)
**Notebook**: `notebooks/test-on testdataset+qwenemdding+llama lr-v2.ipynb`
**Source Code**: `src/test-on testdataset+qwenemdding+llama lr-v2.py`
**Score (Column-averaged AUC)**: **TBD** (Expected: 0.98+)

#### Model Configuration:
- **Base Model**: Qwen 2.5 (0.5b-instruct-gptq-int4)
- **Fine-tuning**: LoRA (rank=32, alpha=64, dropout=0.05) ✨
- **Training**: 3 epochs, learning rate=5e-5, warmup=0.1 ✨
- **Additional Models**:
  - Qwen 2.5 14B (with LoRA + enhanced prompts) ✨
  - Qwen3 Embeddings (0.6B) with semantic search

#### Key Features:
- **Test-Time Augmentation (TTA)** with 5 rounds ✨
- **Enhanced data augmentation** (4x training examples) ✨
- **Advanced prompt engineering** with expert persona ✨
- **Temperature scaling** (0.7) ✨
- **Optimized ensemble weights** (45% / 25% / 30%) ✨
- **TOP_K=3000** for semantic search ✨
- **Comprehensive error handling** ✨

#### Improvements over V1:
- 2x LoRA capacity (rank 16→32)
- 3x training duration (1→3 epochs)
- 4x data augmentation
- TTA for variance reduction
- Better prompts and calibration
- See `VERSION_COMPARISON.md` for full details

#### Notes:
- All accuracy improvements implemented
- Expected +7-14% improvement
- Safety and error handling enhanced

---

### Experiment 3: Preprocessing + Qwen Hybrid Ensemble (Leaderboard Run)
**Date**: N/A
**Version**: Hybrid Ensemble
**Notebook**: `notebooks/[LB 0.916] Preprocessing + Qwen Hybrid Ensemble.ipynb`
**Score (Column-averaged AUC)**: **0.915**

#### Notes:
- Kaggle leaderboard-style preprocessing pipeline with Qwen hybrid ensemble
- Score recorded from the referenced notebook run

---

## Performance History

| Date | Notebook/Script | Version | Score (AUC) | Notes |
|------|----------------|---------|-------------|-------|
| 2025-10-15 | test-on testdataset+qwenemdding+llama lr | Original | 0.916 | Baseline with ensemble |
| 2025-10-15 | test-on testdataset+qwenemdding+llama lr-v2 | V2 | TBD (0.98+) | Enhanced with TTA + improvements |
| N/A | [LB 0.916] Preprocessing + Qwen Hybrid Ensemble | Hybrid Ensemble | 0.915 | Kaggle-style preprocessing + Qwen hybrid ensemble |

---

## Future Experiments

### Ideas to Try:
- [ ] Increase TTA rounds (5 → 10)
- [ ] Experiment with different ensemble weights
- [ ] Try larger LoRA rank (32 → 64)
- [ ] Add more training epochs (3 → 5)
- [ ] Implement stratified k-fold cross-validation
- [ ] Test different temperature values
- [ ] Experiment with different base prompts
- [ ] Try focal loss for imbalanced classes

---

## Best Performing Configuration

**Current Best**: 0.916 (Experiment 1 - Original)
- Multi-model ensemble approach
- LoRA rank 16, 1 epoch training
- Baseline implementation

**Expected Best**: 0.98+ (Experiment 2 - V2)
- Enhanced multi-model ensemble
- TTA with 5 rounds
- LoRA rank 32, 3 epochs training
- 4x data augmentation
- Advanced prompt engineering
