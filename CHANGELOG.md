# Changelog

All notable changes to this project will be documented in this file.

## [2025-10-19 00:00:00 JST] - Experiment 5: Pseudo-Training with Llama 3.2

### Recorded
- Added Experiment 5 to EXPERIMENTS.md: "jigsaw-pseudo-training-llama-3-2-3b-instruct-read.ipynb" scored 0.916 AUC
- Pseudo-training approach using Llama 3.2 3B Instruct model
- Matches baseline performance of other 0.916 AUC experiments

### Notes
- Read-only evaluation version of pseudo-training approach
- Same performance as original Qwen ensemble (Experiment 1)

## [2025-10-18 01:00:00 JST] - Critical Fix: Double-Ranking Bug in FIXED-COMPLETE

### Fixed
- **Critical double-ranking bug** in `test-on testdataset+qwenemdding+llama lr-v2-FIXED-COMPLETE.ipynb`
- Removed rank normalization from `inference.py` that was causing submission format errors
- This bug was causing "Submission Scoring Error" on Kaggle even though notebook executed successfully

### Root Cause
- `inference.py` was applying rank normalization to predictions before saving `submission_qwen.csv`
- Ensemble code was then applying rank normalization AGAIN to already-ranked data
- This double-ranking destroyed the row_id alignment, causing mismatched predictions

### Technical Details
**Before (Buggy)**:
```python
# In inference.py (line 540)
rq = submission['rule_violation'].rank(method='average') / denominator
submission['rule_violation'] = rq  # Pre-ranked data saved

# In ensemble code (line 860)
rq = safe_rank(q['rule_violation'])  # Ranking already-ranked data!
```

**After (Fixed)**:
```python
# In inference.py (removed lines 535-541)
# Save raw logprobs directly, no pre-ranking

# In ensemble code (unchanged)
rq = safe_rank(q['rule_violation'])  # Now ranks raw logprobs only once
```

### Impact
- Notebook will now generate correct submission format
- Row_id values properly aligned with predictions
- Prevents "Submission Scoring Error" on Kaggle platform

### Comparison with Working Version
- The original `[LB 0.916] Preprocessing + Qwen Hybrid Ensemble.ipynb` does NOT pre-rank in individual inference scripts
- It only applies ranking once in the final ensemble step
- FIXED-COMPLETE now follows the same pattern

## [2025-10-18 00:00:00 JST] - Experiment 4 Score Recorded

### Recorded
- Added Experiment 4 to EXPERIMENTS.md: "[LB 0.916] Preprocessing + Qwen Hybrid Ensemble" scored 0.914 AUC
- This is a different run from Experiment 3 (0.915) with enhanced preprocessing features
- Key features: TTA with 4 variants, enhanced text cleaning, LoRA fine-tuned embeddings, rule canonicalization

### Notes
- Slight performance decrease (0.915 → 0.914) possibly due to more aggressive text cleaning
- Version includes `clean-text` library preprocessing and custom emoji/markdown stripping
- Added LoRA fine-tuning for Qwen3 Embeddings model

## [2025-10-17 23:35:52 JST] - Complete Fixed Notebook with Ensemble

### Fixed
- **Missing ensemble pipeline**: Previous FIXED notebook only ran Qwen 0.5B, missing 14B and embeddings models
- **Wrong output filename**: Changed final output from `/kaggle/working/submission.csv` to `submission.csv` (Kaggle expects this exact name in working directory)
- Complete 3-model ensemble now included: Qwen 0.5B + Qwen 14B + Qwen3 Embeddings

### Added
- `src/test-on testdataset+qwenemdding+llama lr-v2-FIXED-COMPLETE.py` - Complete fixed script with all models
- `notebooks/test-on testdataset+qwenemdding+llama lr-v2-FIXED-COMPLETE.ipynb` - Complete fixed notebook

### Root Cause of "Submission CSV Not Found" Error
1. **Incomplete pipeline**: Only ran first model, never created final ensemble
2. **Wrong path**: Saved to `/kaggle/working/submission.csv` instead of `submission.csv` in current directory
3. Kaggle expects `submission.csv` in the notebook's working directory

### Solution
- Added all missing cells from original v2 (Qwen 14B inference, Qwen3 embeddings, ensemble blending)
- Changed output path from `/kaggle/working/submission.csv` to `submission.csv`
- Maintained all row_id preservation fixes from previous version

## [2025-10-17 13:40:08 JST] - Critical Bug Fix: row_id Preservation

### Fixed
- **Critical bug causing Kaggle submission errors**: Fixed `row_id` column being dropped during dataset processing
- Modified `utils.py::build_dataset()` to preserve `row_id` when it exists in test data
- Updated `inference.py` to correctly retrieve `row_id` from processed dataset instead of original dataframe
- This fix resolves the "Submission Scoring Error" that occurred in the v2 notebook

### Added
- `src/test-on testdataset+qwenemdding+llama lr-v2-FIXED.py` - Fixed Python script
- `notebooks/test-on testdataset+qwenemdding+llama lr-v2-FIXED.ipynb` - Fixed notebook

### Root Cause
- The `build_dataset()` function was only keeping `prompt` and `completion` columns, dropping all others including `row_id`
- This caused the inference code to fail when trying to access `df_slice["row_id"].values`
- Working notebooks (LB 0.916 and EDA+TF-IDF) avoided this by not using `build_dataset()` for test data

### Technical Details
- Added logic to detect and preserve `row_id` column before dataset transformation
- Modified both `run_inference_on_device()` and `run_tta_inference()` to use dataset's `row_id` instead of original dataframe
- Ensures compatibility with Kaggle's submission format requirements

## [2025-10-15 22:02:41 JST] - Version Separation: Original vs V2

### Added
- Created V2 versions of code files with all improvements
- `src/test-on testdataset+qwenemdding+llama lr-v2.py` - Improved Python script
- `notebooks/test-on testdataset+qwenemdding+llama lr-v2.ipynb` - Improved notebook
- `VERSION_COMPARISON.md` - Detailed comparison document

### Preserved
- Original files kept intact as baseline (score 0.916)
- `src/test-on testdataset+qwenemdding+llama lr.py` - Original Python script
- `notebooks/test-on testdataset+qwenemdding+llama lr.ipynb` - Original notebook

### Documentation
- Comprehensive comparison table in VERSION_COMPARISON.md
- Side-by-side feature comparison
- Expected performance improvements documented
- Clear usage instructions for both versions

## [2025-10-15 21:57:58 JST] - Experiment Tracking System

### Added
- Created EXPERIMENTS.md to track experiment results
- Documented initial experiment with score 0.916 (column-averaged AUC)
- Added performance history table
- Included future experiment ideas
- Documented best performing configuration

### Structure
- Experiment log with detailed configuration
- Performance history table for quick reference
- Future experiments section for planning
- Best performing configuration tracking

## [2025-10-15 21:50:05 JST] - Accuracy Improvements

### Added
- Test-Time Augmentation (TTA) with 5 rounds of inference using different example combinations
- Enhanced data augmentation: all 4 combinations of positive/negative examples (4x training data)
- Advanced prompt engineering with clearer structure and better context
- Temperature scaling (0.7) for better probability calibration
- Weighted semantic scoring in embedding model
- Enhanced system prompts for 14b model with expert moderator persona

### Optimized
- LoRA rank increased from 16 to 32 for more model capacity
- LoRA alpha increased from 32 to 64
- Training epochs increased from 1 to 3 for better convergence
- Learning rate optimized to 5e-5 (from 1e-4) for stable training
- Warmup ratio increased to 0.1 (from 0.03) for better training stability
- LoRA dropout reduced to 0.05 (from 0.1) for less regularization
- Embedding model TOP_K increased from 2000 to 3000 for better coverage
- Ensemble weights optimized: 0.5b+TTA=45%, embeddings=25%, 14b=30%

### Improved
- Prompt templates with explicit "should answer Yes/No" guidance
- Cross-validation strategy: examples used as training data with alternating pairs
- Duplicate removal strategy preserves context variations
- Better formatting in prompts with clear section labels
- Epoch-based checkpoint saving (save_strategy="epoch")

### Performance Expected
- Higher accuracy from increased model capacity (rank 32 vs 16)
- Better generalization from 3x more training epochs
- Reduced variance through TTA averaging
- Improved calibration from temperature scaling and better ensemble weights

## [2025-10-15 21:43:33 JST] - Safety Improvements

### Added
- Comprehensive error handling throughout all modules
- Input validation for DataFrame columns and required fields
- File existence checks before reading files
- Path validation for all data and model paths
- GPU resource cleanup with `finally` blocks
- Safe ranking with division by zero protection
- Exception handling in multiprocessing workers
- Worker failure detection and reporting
- Fallback to single GPU mode when fewer GPUs are available
- Data integrity validation for predictions and submissions
- Directory creation with `os.makedirs(exist_ok=True)`
- Detailed error messages with traceback logging
- Proper exit codes on failures

### Improved
- All functions now include try-except blocks with proper error messages
- Multiprocessing workers now handle errors gracefully and report failures
- GPU memory cleanup after inference operations
- File I/O operations are now safer with validation
- Better error messages written to stderr for debugging
- Success messages with checkmark indicators
- Model loading with path validation

### Fixed
- Potential division by zero in rank calculations
- Missing error handling in data loading functions
- Unhandled exceptions in worker processes
- Missing validation for empty DataFrames
- Resource leaks from GPU memory
## [2025-10-17 20:27:20 JST] - Accuracy Improvements for LB 0.916 Notebook

### Improved
- Expanded pseudo-labeled augmentation to use the full test set when building LoRA training data and semantic corpus
- Added deterministic multi-example TTA with probability averaging to `inference.py` for stabler Qwen 0.5B predictions
- Refined semantic search scoring with temperature-controlled softmax weighting for more discriminative logits

### Added
- New `SEMANTIC_TEMPERATURE` constant to tune embedding-based probability sharpening

