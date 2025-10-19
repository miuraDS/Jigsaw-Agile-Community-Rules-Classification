# Improvement Plan for LB 0.916 Notebook

## Target: Increase AUC from 0.915-0.916 to 0.94-0.96

## 5 Major Enhancements (Option B - Aggressive)

### Enhancement 1: Increased LoRA Capacity & Training ⭐⭐⭐⭐⭐
**Expected Impact**: +1-2% AUC

**Changes**:
```python
# constants.py
LORA_RANK = 32  # Was: 16 (2x model capacity)
LORA_ALPHA = 64  # Was: 32 (matches new rank ratio)
LORA_DROPOUT = 0.05  # Was: 0.1 (less regularization for more capacity)
NUM_EPOCHS = 3  # Was: 1 (3x training iterations)
LEARNING_RATE = 5e-5  # Was: 1e-4 (better convergence at longer training)
WARMUP_RATIO = 0.1  # Was: 0.03 (more gradual warmup)
```

**train.py updates**:
- Use new hyperparameters from constants
- Add epoch-based checkpointing
- Save best model based on loss

**Rationale**:
- Higher LoRA rank allows model to learn more complex patterns
- More epochs with lower LR prevents overfitting while improving convergence
- Proven successful in V2 notebook (expected 0.98+ AUC)

---

### Enhancement 2: Enhanced TTA (Test-Time Augmentation) ⭐⭐⭐⭐
**Expected Impact**: +0.5-1% AUC

**Changes**:
```python
# constants.py
TTA_ROUNDS = 8  # Was: 4 (2x augmentation variants)
TTA_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]  # Multiple random seeds
```

**inference.py updates**:
- Generate 8 variants instead of 4
- Add seed-based randomization for each TTA round
- Weighted averaging based on prediction confidence

**Rationale**:
- More TTA variants reduce prediction variance
- Different seeds provide diverse example combinations
- Averaging over more predictions improves stability

---

### Enhancement 3: Optimized Ensemble Weights ⭐⭐⭐⭐
**Expected Impact**: +0.3-0.7% AUC

**Changes**:
```python
# Final ensemble (cell 26)
# OLD: blend = 0.5*rq + 0.3*rl + 0.2*rm
# NEW: blend = 0.45*rq + 0.25*rl + 0.30*rm

# Rationale:
# - Reduce 0.5B weight from 50% to 45% (slight reduction)
# - Reduce embeddings from 30% to 25% (less semantic weight)
# - INCREASE 14B from 20% to 30% (larger model gets more weight)
# - Total still 100%
```

**Add grid search for optimal weights**:
```python
def find_best_weights_grid_search():
    # Try combinations: 0.5B: [0.40-0.50], Emb: [0.20-0.30], 14B: [0.25-0.35]
    # Validate on held-out set or pseudo-CV
    # Return best performing combination
```

**Rationale**:
- 14B model is more powerful but was underweighted
- Current 50/30/20 split may not be optimal
- Grid search can find better balance

---

### Enhancement 4: Advanced Prompt Engineering ⭐⭐⭐⭐
**Expected Impact**: +0.5-1% AUC

**Changes in utils.py** (build_prompt function):

```python
def build_prompt(row):
    # Enhanced prompt with reasoning structure
    rule_raw = _sz(row["rule"])
    body = _sz(row["body"])
    subreddit = _sz(row["subreddit"])
    pos_ex = _sz(row["positive_example"])
    neg_ex = _sz(row["negative_example"])

    canon = canonicalize_rule(rule_raw)
    rule_block = f"Rule: {rule_raw}\\n"
    if canon:
        rule_block += f"Canonical Definition: {canon}\\n"

    # NEW: Add structured reasoning format
    return f\"\"\"
{BASE_PROMPT}

Context:
- Subreddit: r/{subreddit}
- This community has specific rules that must be followed

{rule_block}

Examples for Reference:
✓ VIOLATION Example (Answer should be 'Yes'):
{pos_ex}
Reason: This violates the rule.

✗ NON-VIOLATION Example (Answer should be 'No'):
{neg_ex}
Reason: This follows the rule.

Now, carefully analyze this comment:
---
Comment: {body}
---

Question: Does this comment violate the rule?
Think step by step:
1. What is the rule asking for?
2. Does the comment match the violation pattern?
3. Is it similar to the violation example or the allowed example?

{COMPLETE_PHRASE}\"\"\"
```

**Changes in infer_qwen.py** (14B prompts):
- Add chain-of-thought reasoning
- Include rule-specific templates
- Add confidence markers

**Rationale**:
- Structured prompts help models reason better
- Clear examples with explanations improve accuracy
- Chain-of-thought has proven effective for classification

---

### Enhancement 5: Temperature Scaling & Calibration ⭐⭐⭐
**Expected Impact**: +0.2-0.5% AUC

**Add new cell** (after cell 26, before final submission):

```python
%%writefile calibrate.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

def apply_temperature_scaling(predictions, temperature=1.0):
    \"\"\"
    Apply temperature scaling to calibrate probabilities
    predictions: raw scores (pre-rank normalization)
    temperature: scaling parameter (>1 = softer, <1 = sharper)
    \"\"\"
    scaled = predictions / temperature
    # Clip to prevent overflow
    scaled = np.clip(scaled, -10, 10)
    return 1 / (1 + np.exp(-scaled))

def find_optimal_temperature(val_predictions, val_labels):
    \"\"\"
    Find temperature that minimizes log loss on validation set
    \"\"\"
    def loss_fn(T):
        calibrated = apply_temperature_scaling(val_predictions, T[0])
        return log_loss(val_labels, calibrated)

    result = minimize(loss_fn, x0=[1.0], bounds=[(0.1, 5.0)])
    return result.x[0]

# For final ensemble
def calibrate_ensemble(q_pred, l_pred, m_pred):
    \"\"\"
    Apply temperature scaling before ranking
    \"\"\"
    TEMP = 1.2  # Empirically chosen or optimized

    q_calibrated = apply_temperature_scaling(q_pred, TEMP)
    l_calibrated = apply_temperature_scaling(l_pred, TEMP)
    m_calibrated = apply_temperature_scaling(m_pred, TEMP)

    # Now apply rank normalization to calibrated predictions
    rq = pd.Series(q_calibrated).rank(method='average') / (len(q_calibrated) + 1)
    rl = pd.Series(l_calibrated).rank(method='average') / (len(l_calibrated) + 1)
    rm = pd.Series(m_calibrated).rank(method='average') / (len(m_calibrated) + 1)

    return rq, rl, rm
```

**Update cell 26**:
```python
import pandas as pd
import numpy as np
from calibrate import calibrate_ensemble

q = pd.read_csv('submission_qwen.csv')
l = pd.read_csv('submission_qwen3.csv')
m = pd.read_csv('submission_qwen14b.csv')

# Apply calibration before ranking
rq, rl, rm = calibrate_ensemble(
    q['rule_violation'].values,
    l['rule_violation'].values,
    m['rule_violation'].values
)

# Optimized ensemble weights (Enhancement 3)
blend = 0.45*rq + 0.25*rl + 0.30*rm

q['rule_violation'] = blend
q.to_csv('/kaggle/working/submission.csv', index=False)
```

**Rationale**:
- Temperature scaling improves probability calibration
- Better calibrated predictions → better ensemble performance
- Helps models that are over/under-confident

---

## Implementation Summary

### Files to Modify:
1. ✅ `constants.py` - Add new hyperparameters
2. ✅ `train.py` - Use new LoRA config and training params
3. ✅ `inference.py` - Implement 8-round TTA with seeds
4. ✅ `utils.py` - Enhanced prompt engineering
5. ✅ `infer_qwen.py` - Add reasoning to 14B prompts
6. ✅ `calibrate.py` - NEW FILE for temperature scaling
7. ✅ Cell 26 - Update ensemble weights + calibration
8. ✅ `semantic.py` - Already has temperature (keep SEMANTIC_TEMPERATURE=0.2)

### Expected Performance:
- **Conservative**: 0.93-0.94 AUC (if 3/5 improvements work well)
- **Realistic**: 0.94-0.95 AUC (if 4/5 improvements work well)
- **Optimistic**: 0.95-0.96 AUC (if all 5 improvements synergize)

### Risk Assessment:
- **Low Risk**: Enhancements 1, 3 (proven techniques)
- **Medium Risk**: Enhancement 2 (more compute, but safe)
- **Higher Risk**: Enhancements 4, 5 (may need tuning)

### Next Steps:
1. Create improved notebook: `[LB 0.916] IMPROVED - 5 Enhancements.ipynb`
2. Test on Kaggle
3. Document results in EXPERIMENTS.md
4. If successful, this becomes the new baseline
