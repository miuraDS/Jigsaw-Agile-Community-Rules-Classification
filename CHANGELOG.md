# Changelog

All notable changes to this project will be documented in this file.

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
