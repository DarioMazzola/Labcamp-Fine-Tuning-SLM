# train_vision_sagemaker.py
import os
os.environ["UNSLOTH_OFFLINE"] = "1"

import unsloth
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
import argparse
import logging
import sys
from transformers import TextStreamer
import time

import torch
from datasets import load_dataset
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig

# Setup llama.cpp symlink for SageMaker
if not os.path.exists("/opt/ml/code/llama.cpp"):
    os.symlink("/opt/llama.cpp", "/opt/ml/code/llama.cpp")
os.environ["PATH"] = f"/opt/ml/code/llama.cpp:{os.environ['PATH']}"
os.environ["UNSLOTH_SKIP_LLAMA_CPP_INSTALL"] = "1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_vision.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print("Current dir:", os.getcwd())
print("Files here:", os.listdir("."))
print("llama-cli found at:", os.popen("which llama-cli").read())

# ----------------------------
# Supported Vision Models
# ----------------------------
SUPPORTED_VISION_MODELS = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",
    "unsloth/Pixtral-12B-2409-bnb-4bit",
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",
]

# ----------------------------
# Argument Parsing and Validation
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Vision Model Fine-tuning for SageMaker")
    
    # Model and data
    p.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-11B-Vision-Instruct")
    p.add_argument("--dataset_name", type=str, default="unsloth/Radiology_mini")
    p.add_argument("--instruction", type=str, default="You are an expert radiographer. Describe accurately what you see in this image.")
    
    # Training parameters
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None, help="Override num_train_epochs if set")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--max_seq_length", type=int, default=2048)
    
    # Model configuration
    p.add_argument("--load_in_4bit", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--fp16", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", default=True)
    
    # LoRA parameters
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--finetune_vision_layers", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--finetune_language_layers", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--finetune_attention_modules", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--finetune_mlp_modules", type=lambda x: x.lower() == "true", default=True)
    
    # Inference parameters
    p.add_argument("--test_after_training", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--min_p", type=float, default=0.1)
    
    # Hub and output
    p.add_argument("--push_to_hub", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--hub_model_id", type=str, default=None)
    p.add_argument("--output_subdir", type=str, default=".")
    p.add_argument("--max_dataset_size", type=int, default=None)
    
    return p.parse_args()

def validate_args(args):
    """Validate training arguments"""
    errors = []
    
    if args.learning_rate <= 0 or args.learning_rate >= 1:
        errors.append(f"Learning rate should be between 0 and 1, got {args.learning_rate}")
        
    if args.max_seq_length > 4096:
        errors.append(f"Max sequence length too large: {args.max_seq_length}")
        
    if args.per_device_train_batch_size * args.gradient_accumulation_steps > 32:
        logger.warning("Large effective batch size might cause OOM issues")
        
    if args.lora_r > args.lora_alpha:
        logger.warning("LoRA rank higher than alpha, consider adjusting")
        
    if errors:
        raise ValueError("Validation errors: " + "; ".join(errors))

# ----------------------------
# Dataset Conversion
# ----------------------------
def convert_to_conversation(sample, instruction):
    """Convert dataset sample to conversation format"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["caption"]}
            ]
        },
    ]
    return {"messages": conversation}

# ----------------------------
# Inference Testing
# ----------------------------
def test_model_inference(model, tokenizer, dataset, instruction, args):
    """Test model inference after training"""
    try:
        logger.info("=" * 50)
        logger.info("TESTING MODEL INFERENCE")
        logger.info("=" * 50)
        
        FastVisionModel.for_inference(model)
        
        image = dataset[0]["image"]
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        start_time = time.time()
        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            temperature=args.temperature,
            min_p=args.min_p
        )
        end_time = time.time()
        
        logger.info(f"\nInference time: {end_time - start_time:.2f} seconds")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")

# ----------------------------
# Main Training Function
# ----------------------------
def main():
    try:
        args = parse_args()
        validate_args(args)
        
        # SageMaker environment variables
        sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        sm_output_data_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
        output_dir = os.path.join(sm_model_dir, args.output_subdir if args.output_subdir != "." else "")
        
        # Log system info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # HF Hub login
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if args.push_to_hub and hf_token:
            try:
                login(token=hf_token)
                logger.info("Logged in to Hugging Face Hub")
            except Exception as e:
                logger.error(f"Failed to login to Hugging Face Hub: {e}")
                raise
        
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model_name}")
        model, tokenizer = FastVisionModel.from_pretrained(
            args.model_name,
            load_in_4bit=args.load_in_4bit,
            use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        )
        
        # Apply LoRA
        logger.info("Applying LoRA configuration")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=args.finetune_vision_layers,
            finetune_language_layers=args.finetune_language_layers,
            finetune_attention_modules=args.finetune_attention_modules,
            finetune_mlp_modules=args.finetune_mlp_modules,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Load and prepare dataset
        logger.info(f"Loading dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split="train")
        
        if args.max_dataset_size:
            dataset = dataset.select(range(min(len(dataset), args.max_dataset_size)))
            logger.info(f"Limited dataset to {len(dataset)} samples")
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1, seed=3407)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        
        # Convert to conversation format
        logger.info("Converting dataset to conversation format")
        converted_train = [convert_to_conversation(sample, args.instruction) for sample in train_dataset]
        converted_eval = [convert_to_conversation(sample, args.instruction) for sample in eval_dataset]
        
        # Training configuration
        logger.info("Configuring trainer")
        
        # Build training args dynamically to avoid None values
        training_config = {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_steps": args.warmup_steps,
            "learning_rate": args.learning_rate,
            "fp16": args.fp16 and not is_bfloat16_supported(),
            "bf16": args.bf16 and is_bfloat16_supported(),
            "logging_steps": args.logging_steps,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "eval_strategy": "steps",
            "eval_steps": args.eval_steps,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "seed": 3407,
            "output_dir": output_dir,
            "report_to": "none",
            # Vision-specific settings
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "max_length": args.max_seq_length,
        }
        
        # Add max_steps OR num_train_epochs (not both)
        if args.max_steps and args.max_steps > 0:
            training_config["max_steps"] = args.max_steps
            logger.info(f"Using max_steps: {args.max_steps}")
        else:
            training_config["num_train_epochs"] = args.num_train_epochs
            logger.info(f"Using num_train_epochs: {args.num_train_epochs}")
        
        training_args = SFTConfig(**training_config)
        
        # Create trainer
        FastVisionModel.for_training(model)
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=converted_train,
            eval_dataset=converted_eval,
            args=training_args,
        )
        
        # Show GPU stats
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.info(f"{start_gpu_memory} GB of memory reserved.")
        
        # Train
        logger.info("Starting training...")
        trainer_stats = trainer.train()
        logger.info("Training completed successfully!")
        
        # Show final stats
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            logger.info(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
            logger.info(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
            logger.info(f"Peak reserved memory = {used_memory} GB.")
            logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
            logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        # Log training metrics
        logger.info(f"Final training loss: {trainer_stats.training_loss:.4f}")
        
        # Test inference if requested
        if args.test_after_training:
            test_model_inference(model, tokenizer, dataset["train"], args.instruction, args)
        
        # Final evaluation
        eval_result = trainer.evaluate()
        logger.info(f"Final evaluation loss: {eval_result['eval_loss']:.4f}")
        
        # Save model
        logger.info("Saving model and artifacts...")
        logger.info("Merging LoRA weights into base model...")
        model.save_pretrained_merged(output_dir, tokenizer)
        
        # Save in GGUF format
        try:
            logger.info("Saving model in GGUF format...")
            model.save_pretrained_gguf(
                output_dir,
                tokenizer,
                quantization_method="f16"
            )
            logger.info(f"All artifacts saved to: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save model in GGUF format: {e}")
        
        # Push to hub if requested
        if args.push_to_hub and args.hub_model_id:
            logger.info(f"Pushing model to Hub: {args.hub_model_id}")
            model.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

