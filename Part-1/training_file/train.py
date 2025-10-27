# train_sagemaker.py
import os
os.environ["UNSLOTH_OFFLINE"] = "1"

import unsloth
from unsloth import FastModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import argparse
import logging
import sys
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import os, glob, shutil, logging
import torch
from datasets import load_dataset
from huggingface_hub import login
from trl import SFTTrainer

if not os.path.exists("/opt/ml/code/llama.cpp"):
    os.symlink("/opt/llama.cpp", "/opt/ml/code/llama.cpp")
os.environ["PATH"] = f"/opt/ml/code/llama.cpp:{os.environ['PATH']}"
os.environ["UNSLOTH_SKIP_LLAMA_CPP_INSTALL"] = "1"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print("Current dir:", os.getcwd())
print("Files here:", os.listdir("."))
print("llama-cli found at:", os.popen("which llama-cli").read())


# ----------------------------
# Argomenti dell'esecuzione con validazione
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Modello e dati
    p.add_argument("--model_name", type=str, default="unsloth/gemma-3-4b-it")
    p.add_argument("--dataset_name", type=str, default="Jofthomas/hermes-function-calling-thinking-V1")
    p.add_argument("--use_hub_dataset", type=lambda x: x.lower() == "true", default=True, help="Se true, usa load_dataset(<dataset_name>)")
    
    # SFT / training - Valori più conservativi
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=500)
    
    # Precisione e quantizzazione
    p.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--fp16", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--dtype", type=str, default=None, help="Data type for model (None for auto)")
    p.add_argument("--load_in_8bit", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--load_in_4bit", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--packing", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", default=True)

    # LoRA / PEFT - Parametri ottimizzati
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Hub e monitoring
    p.add_argument("--push_to_hub", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--hub_model_id", type=str, default=None)
    
    # Output
    p.add_argument("--output_subdir", type=str, default=".")
    p.add_argument("--max_dataset_size", type=int, default=None, help="Limit dataset size for testing")
    
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
    
def move_gguf_and_modelfiles(output_dir: str, logger: logging.Logger) -> None:
    """
    Sposta i file .gguf e Modelfile dalla directory corrente (CWD)
    nella cartella di output specificata (output_dir).

    Args:
        output_dir (str): Cartella di destinazione, es. /opt/ml/model
        logger (logging.Logger): Istanza del logger per messaggi informativi
    """
    try:
        cwd = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Current working directory (CWD): {cwd}")
        logger.info(f"Target output directory: {output_dir}")

        # Cerca i file da spostare
        gguf_files = glob.glob(os.path.join(cwd, "*.gguf"))
        modelfiles = glob.glob(os.path.join(cwd, "Modelfile"))
        all_files = gguf_files + modelfiles

        if not all_files:
            logger.warning("Nessun file .gguf o Modelfile trovato nella CWD.")
            return

        for src in all_files:
            dst = os.path.join(output_dir, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                logger.info(f"Moving {src} -> {dst}")
                shutil.move(src, dst)

        logger.info("Final list of files in output_dir:")
        for f in sorted(os.listdir(output_dir)):
            logger.info(f" - {f}")

    except Exception as e:
        logger.error(f"Errore durante lo spostamento dei file .gguf/Modelfile: {e}")
        raise


def main():
    try:
        args = parse_args()
        validate_args(args)
        
        # SageMaker environment variables (deve essere prima di output_dir)
        sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        sm_output_data_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
        
        # Log system info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        output_dir = os.path.join(sm_model_dir, args.output_subdir if args.output_subdir != "." else "")

        # HF Hub login
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if args.push_to_hub and hf_token:
            try:
                login(token=hf_token)
                logger.info("Logged in to Hugging Face Hub")
            except Exception as e:
                logger.error(f"Failed to login to Hugging Face Hub: {e}")
                raise

        # Setup model and tokenizer
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
        )

        model = FastModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "gemma-3",
        )

        # Caricamento dataset
        if args.use_hub_dataset:
            dataset = load_dataset(args.dataset_name, split="train")
        else:
            dataset = load_dataset("rewoo/planner_instruction_tuning_2k", split="train")
            
        # Limita dimensione dataset se specificato
        if args.max_dataset_size:
            dataset = dataset.select(range(min(len(dataset), args.max_dataset_size)))
            
        dataset = dataset.train_test_split(test_size=0.1, seed=3407)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        def formatting_prompts_func(examples):
            """Converte il dataset in formato conversazionale Gemma-3"""
            texts = []
            
            for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
                # Costruisci il prompt utente
                if inp.strip():
                    user_content = f"{instr}\n\nInput: {inp}"
                else:
                    user_content = instr
                
                # Formato conversazionale
                conversation = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": out}
                ]

                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
            
            return {"text": texts}

        train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, remove_columns=eval_dataset.column_names)

        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16 and not is_bfloat16_supported(),
            bf16=args.bf16 and is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type=args.lr_scheduler_type,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            gradient_checkpointing=args.gradient_checkpointing,
        )

        # Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
            dataset_num_proc=2,
            packing=args.packing,
            args=training_args,
        )

        # TRAIN ON RESPONSES ONLY
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        
        # Training
        logger.info("Starting training...")
        trainer_stats = trainer.train()
        logger.info("Training completed successfully!")
        
        # Log training metrics
        logger.info(f"Final training loss: {trainer_stats.training_loss:.4f}")

        # Save model and artifacts
        logger.info("Saving model and artifacts...")
        
        # SALVA IL MODELLO FUSO
        logger.info("Merging LoRA weights into base model...")
        model.save_pretrained_merged(output_dir, tokenizer)

        # Esporta in GGUF (GGUF = formato llama.cpp)
        logger.info("Saving model in GGUF format...")
        model.save_pretrained_gguf(
            output_dir,              # cartella HF (Hugging Face) con config.json
            tokenizer,
            quantization_method="f16"  # es.: "q4_k_m", "q8_0", "f16"
        )

        # Sposta eventuali file generati nella CWD dentro output_dir
        move_gguf_and_modelfiles(output_dir, logger)

        logger.info(f"All artifacts saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()