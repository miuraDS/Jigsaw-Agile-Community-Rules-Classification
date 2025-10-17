# How to Run on Kaggle

This document explains how to run the different versions on Kaggle notebooks.

## ⚠️ Important: File Structure Issue

The files `src/test-on testdataset+qwenemdding+llama lr.py` and `src/test-on testdataset+qwenemdding+llama lr-v2.py` **CANNOT be run in a single Kaggle cell** because they use `%%writefile` magic commands.

## 🎯 Three Ways to Run on Kaggle

### Option 1: Use the Notebook Files (Recommended)
✅ **Best for Kaggle**

Upload the `.ipynb` files directly to Kaggle:
- `notebooks/test-on testdataset+qwenemdding+llama lr.ipynb` (Original)
- `notebooks/test-on testdataset+qwenemdding+llama lr-v2.ipynb` (V2)

These are already properly formatted as Jupyter notebooks with separate cells.

**Steps:**
1. Go to Kaggle → New Notebook
2. File → Upload Notebook
3. Select the `.ipynb` file
4. Run cells in sequence

---

### Option 2: Use Single-Cell Version (Easiest)
✅ **Best for quick testing**

Use the single-cell runnable file:
```python
# Copy contents of:
src/test-on testdataset+qwenemdding+llama lr-v2-single-cell.py
```

**Steps:**
1. Create new Kaggle notebook
2. Paste entire contents into a single cell
3. Run the cell
4. Check `/kaggle/working/submission.csv`

**Features:**
- ✅ All code in one file
- ✅ No `%%writefile` magic commands
- ✅ Can run in single cell
- ✅ Includes TTA and ensemble
- ⚠️ Training is commented out by default (uncomment if needed)

---

### Option 3: Split into Cells Manually
⚠️ **More work but gives full control**

Copy the `.py` file and split at `# --- new cell ---` markers:

1. Create new Kaggle notebook
2. Copy content between markers into separate cells
3. Remove `%%writefile` prefixes
4. Run cells in sequence

---

## 📊 Comparison of Options

| Option | Ease | Control | Time |
|--------|------|---------|------|
| Upload .ipynb | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Instant |
| Single-cell .py | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~5 min |
| Manual split | ⭐⭐ | ⭐⭐⭐⭐⭐ | ~15 min |

---

## 🔧 What Each Version Does

### Original (0.916 AUC)
**Files:**
- `notebooks/test-on testdataset+qwenemdding+llama lr.ipynb`

**Workflow:**
1. Writes `constants.py`, `utils.py`, `train.py`, etc.
2. Trains 0.5B model (1 epoch, rank=16)
3. Runs inference with 0.5B
4. Runs inference with 14B
5. Runs semantic search with embeddings
6. Blends all three (50/30/20)
7. Outputs `/kaggle/working/submission.csv`

---

### V2 Improved (Expected 0.98+ AUC)
**Files:**
- `notebooks/test-on testdataset+qwenemdding+llama lr-v2.ipynb`
- `src/test-on testdataset+qwenemdding+llama lr-v2-single-cell.py` ⭐

**Workflow:**
1. Writes enhanced versions of Python files
2. Trains 0.5B model (3 epochs, rank=32) ⬆️
3. Runs **TTA inference** with 5 rounds ⬆️
4. Runs enhanced 14B inference with better prompts ⬆️
5. Runs semantic search (TOP_K=3000) ⬆️
6. Blends with optimized weights (45/25/30) ⬆️
7. Outputs `/kaggle/working/submission.csv`

**Single-cell version:**
- Skips file writing
- All code in one executable
- Training commented out (uncomment if needed)
- Embeddings commented out (for speed)
- Uses 60/40 blend (0.5B+TTA / 14B) without embeddings

---

## 💡 Recommended Usage

### For Best Results:
```python
# Use V2 notebook file
notebooks/test-on testdataset+qwenemdding+llama lr-v2.ipynb
```

### For Quick Testing:
```python
# Use single-cell version
src/test-on testdataset+qwenemdding+llama lr-v2-single-cell.py
```

### For Experimentation:
```python
# Modify V2 notebook and adjust parameters
```

---

## 🐛 Troubleshooting

### Issue: "%%writefile" not recognized
**Cause:** Trying to run `.py` file in single cell
**Solution:** Use the `-single-cell.py` version or upload `.ipynb`

### Issue: "File not found" errors
**Cause:** Required model files missing
**Solution:** Add as Kaggle dataset inputs:
- Qwen 2.5 models
- LoRA weights
- Training data

### Issue: Out of memory
**Cause:** TTA or embedding model
**Solution:**
- Comment out `generate_submission_embeddings()` in single-cell version
- Reduce `TTA_ROUNDS` from 5 to 3
- Use smaller batch sizes

### Issue: Training takes too long
**Cause:** 3 epochs with large dataset
**Solution:**
- Comment out `train_model()` if model already trained
- Reduce `NUM_EPOCHS` from 3 to 1
- Use pre-trained LoRA weights

---

## 📝 Quick Start Template

### Kaggle Notebook Cell 1:
```python
# Upload the single-cell file or copy-paste:
# src/test-on testdataset+qwenemdding+llama lr-v2-single-cell.py

# The file will:
# 1. Train model (if uncommented)
# 2. Generate predictions with TTA
# 3. Create ensemble
# 4. Save to /kaggle/working/submission.csv
```

### Kaggle Notebook Cell 2:
```python
# Check the submission
import pandas as pd
submission = pd.read_csv('/kaggle/working/submission.csv')
print(submission.head())
print(f"Shape: {submission.shape}")
print(f"Score range: {submission['rule_violation'].min():.4f} - {submission['rule_violation'].max():.4f}")
```

---

## ✅ Expected Output

When successfully run, you should see:
```
============================================================
JIGSAW AGILE COMMUNITY RULES CLASSIFICATION - V2
============================================================
============================================================
GENERATING QWEN 0.5B SUBMISSION (with TTA)
============================================================
TTA Round 1/5
...
✅ Saved submission_qwen.csv
============================================================
GENERATING QWEN 14B SUBMISSION
============================================================
...
✅ Saved submission_qwen14b.csv
============================================================
CREATING ENSEMBLE
============================================================
✅ Final submission saved to /kaggle/working/submission.csv
Ensemble weights: 0.5b+TTA=60%, 14b=40%
Preview:
...
============================================================
✅ ALL DONE! Check /kaggle/working/submission.csv
============================================================
```

The final submission will be at `/kaggle/working/submission.csv` ready to submit!
