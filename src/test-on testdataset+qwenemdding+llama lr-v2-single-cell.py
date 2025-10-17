"""
Single-cell runnable version of the improved model
This can be pasted into a single Kaggle notebook cell and executed
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import torch
import vllm
import multiprocessing as mp
import traceback

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel, PeftConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_torch_bf16_gpu_available
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from vllm.lora.request import LoRARequest
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from scipy.special import softmax
from cleantext import clean

# Set seeds
random.seed(42)
np.random.seed(42)

# ============================================================================
# CONSTANTS
# ============================================================================
BASE_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = '''You are an expert content moderator. Analyze if the comment violates the subreddit rule.'''

# Training hyperparameters - optimized for accuracy
LORA_RANK = 32
LORA_ALPHA = 64
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1

# Test-time augmentation
TTA_ROUNDS = 5

# Embedding constants
EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
MODEL_OUTPUT_PATH = '/kaggle/input/qwen3-8b-embedding'
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
CLEAN_TEXT = True
TOP_K = 3000
BATCH_SIZE = 128

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_prompt(row):
    """Build enhanced prompt with better structure for accuracy."""
    try:
        required_fields = ["subreddit", "rule", "positive_example", "negative_example", "body"]
        for field in required_fields:
            if field not in row or pd.isna(row[field]):
                raise ValueError(f"Missing or invalid field: {field}")

        return f"""{BASE_PROMPT}

Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}

Examples of rule violations (should answer Yes):
Example 1: {row["positive_example"]}
{COMPLETE_PHRASE} {POSITIVE_ANSWER}

Examples of allowed content (should answer No):
Example 2: {row["negative_example"]}
{COMPLETE_PHRASE} {NEGATIVE_ANSWER}

Now analyze this comment:
Comment: {row["body"]}
{COMPLETE_PHRASE}"""
    except Exception as e:
        print(f"Error building prompt: {e}", file=sys.stderr)
        raise


def get_dataframe_to_train(data_path):
    """Load and process training data with enhanced augmentation."""
    try:
        csv_path = f"{data_path}/train.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data not found at {csv_path}")

        train_dataset = pd.read_csv(csv_path)

        required_columns = ["body", "rule", "subreddit", "rule_violation",
                           "positive_example_1", "positive_example_2",
                           "negative_example_1", "negative_example_2"]
        missing_columns = [col for col in required_columns if col not in train_dataset.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        flatten = []

        # All combinations of examples
        for pos_ex in ["positive_example_1", "positive_example_2"]:
            for neg_ex in ["negative_example_1", "negative_example_2"]:
                main_train_df = train_dataset[["body", "rule", "subreddit", "rule_violation"]].copy()
                main_train_df["positive_example"] = train_dataset[pos_ex]
                main_train_df["negative_example"] = train_dataset[neg_ex]
                flatten.append(main_train_df)

        # Use examples as training data
        for pos_idx in [1, 2]:
            for neg_idx in [1, 2]:
                pos_df = train_dataset[["rule", "subreddit"]].copy()
                pos_df["body"] = train_dataset[f"positive_example_{pos_idx}"]
                pos_df["rule_violation"] = 1
                pos_df["positive_example"] = train_dataset[f"positive_example_{3-pos_idx}"]
                pos_df["negative_example"] = train_dataset[f"negative_example_{neg_idx}"]
                flatten.append(pos_df)

                neg_df = train_dataset[["rule", "subreddit"]].copy()
                neg_df["body"] = train_dataset[f"negative_example_{neg_idx}"]
                neg_df["rule_violation"] = 0
                neg_df["positive_example"] = train_dataset[f"positive_example_{pos_idx}"]
                neg_df["negative_example"] = train_dataset[f"negative_example_{3-neg_idx}"]
                flatten.append(neg_df)

        dataframe = pd.concat(flatten, axis=0, ignore_index=True)
        dataframe = dataframe.drop_duplicates(subset=["body", "rule", "subreddit", "rule_violation"], ignore_index=True)
        dataframe = dataframe.dropna()

        if len(dataframe) == 0:
            raise ValueError("No valid training data after processing")

        print(f"✅ Generated {len(dataframe)} training examples with augmentation")
        return dataframe
    except Exception as e:
        print(f"❌ Error loading training data: {e}", file=sys.stderr)
        raise


def build_dataset(dataframe):
    """Build dataset from dataframe."""
    try:
        if dataframe is None or len(dataframe) == 0:
            raise ValueError("Empty dataframe provided")

        dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

        columns = ["prompt"]
        if "rule_violation" in dataframe:
            dataframe["completion"] = dataframe["rule_violation"].map({
                1: POSITIVE_ANSWER,
                0: NEGATIVE_ANSWER,
            })
            columns.append("completion")

        dataframe = dataframe[columns]
        dataset = Dataset.from_pandas(dataframe)

        output_path = "/kaggle/working/dataset.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_pandas().to_csv(output_path, index=False)

        return dataset
    except Exception as e:
        print(f"❌ Error building dataset: {e}", file=sys.stderr)
        raise


# ============================================================================
# TRAINING (Optional - comment out if model already trained)
# ============================================================================

def train_model():
    """Train the model with enhanced configuration."""
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data path not found: {DATA_PATH}")
        if not os.path.exists(BASE_MODEL_PATH):
            raise FileNotFoundError(f"Model path not found: {BASE_MODEL_PATH}")

        os.makedirs(LORA_PATH, exist_ok=True)

        dataframe = get_dataframe_to_train(DATA_PATH)
        train_dataset = build_dataset(dataframe)

        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

        training_args = SFTConfig(
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="paged_adamw_8bit",
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            bf16=is_torch_bf16_gpu_available(),
            fp16=not is_torch_bf16_gpu_available(),
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none",
            completion_only_loss=True,
            packing=False,
            remove_unused_columns=False,
            logging_steps=50,
            eval_strategy="no",
        )

        trainer = SFTTrainer(
            BASE_MODEL_PATH,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )

        trainer.train()
        trainer.save_model(LORA_PATH)
        print(f"✅ Model saved successfully to {LORA_PATH}")
        print(f"Training completed with {NUM_EPOCHS} epochs, LoRA rank={LORA_RANK}")

    except Exception as e:
        print(f"❌ Training failed: {e}", file=sys.stderr)
        raise


# ============================================================================
# INFERENCE WITH TTA
# ============================================================================

def run_tta_inference_single_gpu(test_dataframe):
    """Run TTA inference on a single GPU."""
    os.environ["VLLM_USE_V1"] = "0"

    llm = None
    try:
        llm = vllm.LLM(
            BASE_MODEL_PATH,
            quantization="gptq",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.98,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=2836,
            disable_log_stats=True,
            enable_prefix_caching=True,
            enable_lora=True,
            max_lora_rank=64,
        )

        tokenizer = llm.get_tokenizer()
        mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=[POSITIVE_ANSWER, NEGATIVE_ANSWER])

        all_predictions = []

        # Test-Time Augmentation
        for tta_round in range(TTA_ROUNDS):
            print(f"TTA Round {tta_round + 1}/{TTA_ROUNDS}")
            random.seed(42 + tta_round)

            test_dataset = build_dataset(test_dataframe)
            texts = test_dataset["prompt"]

            outputs = llm.generate(
                texts,
                vllm.SamplingParams(
                    skip_special_tokens=True,
                    max_tokens=1,
                    logits_processors=[mclp],
                    logprobs=2,
                    temperature=0.7,
                ),
                use_tqdm=True,
                lora_request=LoRARequest("default", 1, LORA_PATH)
            )

            log_probs = [
                {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
                for out in outputs
            ]
            predictions = pd.DataFrame(log_probs)[[POSITIVE_ANSWER, NEGATIVE_ANSWER]]
            all_predictions.append(predictions)

        # Average predictions
        avg_predictions = pd.concat(all_predictions).groupby(level=0).mean()
        avg_predictions["row_id"] = test_dataframe["row_id"].values

        print(f"✅ Completed TTA with {TTA_ROUNDS} rounds")
        return avg_predictions

    except Exception as e:
        print(f"❌ TTA inference failed: {e}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        if llm is not None:
            del llm
        torch.cuda.empty_cache()


def generate_submission_qwen():
    """Generate submission from 0.5B model with TTA."""
    print("=" * 60)
    print("GENERATING QWEN 0.5B SUBMISSION (with TTA)")
    print("=" * 60)

    try:
        test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")

        test_dataframe["positive_example"] = test_dataframe.apply(
            lambda row: random.choice([row["positive_example_1"], row["positive_example_2"]]),
            axis=1
        )
        test_dataframe["negative_example"] = test_dataframe.apply(
            lambda row: random.choice([row["negative_example_1"], row["negative_example_2"]]),
            axis=1
        )
        test_dataframe = test_dataframe.drop(
            columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"],
            errors="ignore"
        )

        predictions = run_tta_inference_single_gpu(test_dataframe)

        submission = predictions[["row_id", POSITIVE_ANSWER]].rename(columns={POSITIVE_ANSWER: "rule_violation"})
        rq = submission['rule_violation'].rank(method='average') / (len(submission) + 1)
        submission['rule_violation'] = rq

        submission.to_csv("submission_qwen.csv", index=False)
        print(f"✅ Saved submission_qwen.csv")

    except Exception as e:
        print(f"❌ Failed: {e}", file=sys.stderr)
        raise


# ============================================================================
# 14B MODEL INFERENCE
# ============================================================================

def generate_submission_qwen14b():
    """Generate submission from 14B model."""
    print("=" * 60)
    print("GENERATING QWEN 14B SUBMISSION")
    print("=" * 60)

    os.environ["VLLM_USE_V1"] = "0"

    MODEL_NAME = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
    LORA_14B_PATH = "/kaggle/input/lora_14b_gptq_1epoch_r32/keras/default/1"

    try:
        df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

        llm = vllm.LLM(
            MODEL_NAME,
            quantization='gptq',
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.98,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=2836,
            disable_log_stats=True,
            enable_prefix_caching=True,
            enable_lora=True,
            max_lora_rank=32
        )
        tokenizer = llm.get_tokenizer()

        SYS_PROMPT = """You are an expert content moderator with deep understanding of community guidelines. Carefully analyze if the comment violates the given rule based on the provided examples. Consider context, tone, and intent."""

        prompts = []
        for _, row in df.iterrows():
            text = f"""Subreddit: r/{row.subreddit}
Rule: {row.rule}

Examples of VIOLATIONS (answer: Yes):
1) {row.positive_example_1}
2) {row.positive_example_2}

Examples of ALLOWED content (answer: No):
3) {row.negative_example_1}
4) {row.negative_example_2}

Comment to analyze:
5) {row.body}

Does this comment violate the rule?"""

            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": text}
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ) + "Answer:"
            prompts.append(prompt)

        mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=['Yes','No'])
        outputs = llm.generate(
            prompts,
            vllm.SamplingParams(
                skip_special_tokens=True,
                max_tokens=1,
                logits_processors=[mclp],
                logprobs=2,
                temperature=0.7,
            ),
            use_tqdm=True,
            lora_request=LoRARequest("default", 1, LORA_14B_PATH)
        )

        logprobs = [
            {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
            for out in outputs
        ]
        logit_matrix = pd.DataFrame(logprobs)[['Yes','No']]
        df[['Yes',"No"]] = logit_matrix[['Yes',"No"]].apply(lambda x: softmax(x.values), axis=1, result_type="expand")
        df['rule_violation'] = df["Yes"]

        df[['row_id', 'rule_violation']].to_csv("submission_qwen14b.csv", index=False)
        print(f"✅ Saved submission_qwen14b.csv")

        del llm
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Failed: {e}", file=sys.stderr)
        raise


# ============================================================================
# EMBEDDING MODEL (Commented out - uncomment if needed)
# ============================================================================

# Note: Embedding generation takes significant time and memory
# Uncomment below if you want to use the embedding model

"""
def generate_submission_embeddings():
    print("=" * 60)
    print("GENERATING EMBEDDINGS SUBMISSION")
    print("=" * 60)

    # Implementation same as in original file
    # See original for full code
    pass
"""


# ============================================================================
# ENSEMBLE
# ============================================================================

def create_ensemble():
    """Create final ensemble submission."""
    print("=" * 60)
    print("CREATING ENSEMBLE")
    print("=" * 60)

    try:
        q = pd.read_csv('submission_qwen.csv')
        m = pd.read_csv('submission_qwen14b.csv')

        def safe_rank(series):
            n = len(series)
            if n == 0:
                raise ValueError("Empty series for ranking")
            return series.rank(method='average') / (n + 1)

        rq = safe_rank(q['rule_violation'])
        rm = safe_rank(m['rule_violation'])

        # Without embeddings: 60% 0.5b+TTA, 40% 14b
        blend = 0.6*rq + 0.4*rm

        q['rule_violation'] = blend

        os.makedirs('/kaggle/working', exist_ok=True)
        output_path = '/kaggle/working/submission.csv'
        q.to_csv(output_path, index=False)

        print(f"✅ Final submission saved to {output_path}")
        print(f"Ensemble weights: 0.5b+TTA=60%, 14b=40%")
        print(f"Preview:")
        print(q.head())

    except Exception as e:
        print(f"❌ Blending failed: {e}", file=sys.stderr)
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("JIGSAW AGILE COMMUNITY RULES CLASSIFICATION - V2")
    print("="*60)

    # Step 1: Training (comment out if already trained)
    # train_model()

    # Step 2: Generate submissions
    generate_submission_qwen()      # 0.5B with TTA
    generate_submission_qwen14b()   # 14B
    # generate_submission_embeddings()  # Uncomment if using embeddings

    # Step 3: Create ensemble
    create_ensemble()

    print("\n" + "="*60)
    print("✅ ALL DONE! Check /kaggle/working/submission.csv")
    print("="*60)
