import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (loaded from environment when available)
REGION = os.getenv("AWS_REGION", "us-east-1")
ROLE = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::<ACCOUNT-ID>:role/<ROLE-NAME>")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.p3.2xlarge")
VOLUME_SIZE = int(os.getenv("VOLUME_SIZE", "150"))
IMAGE_NAME = os.getenv("ECR_IMAGE_NAME", "docker-image-name")  # e.g., unsloth/unsloth

# Path to training code directory
SOURCE_DIR = "./training_file"

def get_image_uri():
    """Get the ECR image URI"""
    try:
        account = boto3.client("sts").get_caller_identity()["Account"]
        return f"{account}.dkr.ecr.{REGION}.amazonaws.com/{IMAGE_NAME}:latest"
    except Exception as e:
        logger.error(f"Failed to get account ID: {e}")
        raise

def create_job_name():
    """Create a unique job name with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"vision-finetuning-{timestamp}"

def get_hyperparameters_vision():
    """Get hyperparameters for vision model fine-tuning"""
    hyperparameters = {
        # Model and data
        "model_name": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "dataset_name": "unsloth/Radiology_mini",
        "instruction": "You are an expert radiographer. Describe accurately what you see in this image.",
        # "max_dataset_size": "50",  # Uncomment to limit dataset size for testing
        
        # Training parameters
        # "num_train_epochs": "1",
        "max_steps": "30",  # Uncomment to use max_steps instead of num_train_epochs
        "per_device_train_batch_size": "2",
        "gradient_accumulation_steps": "4",
        "learning_rate": "2e-4",
        "warmup_steps": "5",
        "max_seq_length": "2048",
        
        # Evaluation and saving
        "eval_steps": "50",
        "logging_steps": "1",
        "save_steps": "100",
        "save_strategy": "steps",
        
        # Optimization
        "bf16": "true",
        "fp16": "false",
        "load_in_4bit": "true",
        "gradient_checkpointing": "true",
        
        # LoRA parameters
        "lora_r": "16",
        "lora_alpha": "16",
        "lora_dropout": "0.0",
        
        # Vision-specific fine-tuning
        "finetune_vision_layers": "true",
        "finetune_language_layers": "true",
        "finetune_attention_modules": "true",
        "finetune_mlp_modules": "true",
        
        # Inference testing
        "test_after_training": "true",
        "max_new_tokens": "128",
        "temperature": "1.5",
        "min_p": "0.1",
        
        # Hub (optional)
        "push_to_hub": "false",
        # "hub_model_id": "your-username/model-name",
    }
    
    return hyperparameters

def validate_setup():
    """Validate the setup before starting the job"""
    try:
        # Check AWS credentials
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        logger.info(f"Using AWS Account: {identity['Account']}")
        
        # Check source directory
        if not os.path.exists(SOURCE_DIR):
            logger.error(f"Source directory not found: {SOURCE_DIR}")
            raise FileNotFoundError(f"Create {SOURCE_DIR} and add your training script there")
        
        # Check for vision training script
        train_vision_py = os.path.join(SOURCE_DIR, "train_vision.py")
        if not os.path.exists(train_vision_py):
            logger.warning(f"train_vision.py not found in {SOURCE_DIR}")
            logger.info(f"Available files: {os.listdir(SOURCE_DIR)}")
            logger.info("Make sure to copy finetuning_vision.py to training_file/train_vision.py")
        
        logger.info(f"Source directory validated: {SOURCE_DIR}")
        logger.info(f"Files: {os.listdir(SOURCE_DIR)}")
        
        # Check if role exists
        iam = boto3.client("iam")
        try:
            iam.get_role(RoleName=ROLE.split("/")[-1])
            logger.info("SageMaker execution role validated")
        except Exception as e:
            logger.warning(f"Could not validate role: {e}")
        
        # Check ECR repository
        ecr = boto3.client("ecr", region_name=REGION)
        try:
            ecr.describe_repositories(repositoryNames=[IMAGE_NAME])
            logger.info("ECR repository found")
        except Exception as e:
            logger.warning(f"Could not validate ECR repository: {e}")
            
    except Exception as e:
        logger.error(f"Setup validation failed: {e}")
        raise

def setup_environment():
    """Setup environment variables if needed"""
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        logger.info("HuggingFace token found in environment")
    else:
        logger.warning("HUGGINGFACE_HUB_TOKEN not found in environment")
    
    wandb_token = os.getenv("WANDB_API_KEY")
    if wandb_token:
        logger.info("WandB token found in environment")

def create_estimator():
    """Create and configure the SageMaker estimator for vision fine-tuning"""
    boto3_session = boto3.session.Session(region_name=REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)
    
    hyperparameters = get_hyperparameters_vision()
    image_uri = get_image_uri()
    job_name = create_job_name()
    
    logger.info(f"Creating estimator with job name: {job_name}")
    logger.info(f"Instance type: {INSTANCE_TYPE}")
    logger.info(f"Image URI: {image_uri}")
    logger.info(f"Source directory: {SOURCE_DIR}")
    
    # Environment variables to pass to the container
    environment = {}
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        environment["HUGGINGFACE_HUB_TOKEN"] = hf_token
    
    wandb_token = os.getenv("WANDB_API_KEY")
    if wandb_token:
        environment["WANDB_API_KEY"] = wandb_token
    
    estimator = Estimator(
        image_uri=image_uri,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        volume_size=VOLUME_SIZE,
        max_run=24 * 60 * 60,  # 24 hours
        hyperparameters=hyperparameters,
        environment=environment,
        sagemaker_session=sagemaker_session,
        base_job_name="vision-finetuning",
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {"Name": "train_loss", "Regex": "train_loss: ([0-9\\.]+)"},
            {"Name": "eval_loss", "Regex": "eval_loss: ([0-9\\.]+)"},
            {"Name": "learning_rate", "Regex": "learning_rate: ([0-9\\.]+)"},
            {"Name": "train_runtime", "Regex": "train_runtime: ([0-9\\.]+)"},
            {"Name": "peak_memory_gb", "Regex": "Peak reserved memory = ([0-9\\.]+) GB"},
        ],
        tags=[
            {"Key": "Project", "Value": "VisionFineTuning"},
            {"Key": "Model", "Value": "Llama-3.2-11B-Vision"},
            {"Key": "Environment", "Value": "Development"},
            {"Key": "Type", "Value": "Vision"},
        ],
        source_dir=SOURCE_DIR,
        entry_point="train_vision.py",  # Vision training script
    )
    
    return estimator

def save_job_config(estimator, output_file="job_config_vision.json"):
    """Save job configuration for reference"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "job_type": "vision_finetuning",
        "instance_type": INSTANCE_TYPE,
        "volume_size": VOLUME_SIZE,
        "hyperparameters": estimator.hyperparameters(),
        "role": ROLE,
        "image_uri": estimator.image_uri,
        "source_dir": SOURCE_DIR,
        "entry_point": "train_vision.py",
    }
    
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Job configuration saved to {output_file}")

def main():
    """Main execution function"""
    try:
        logger.info("=" * 70)
        logger.info("STARTING VISION MODEL FINE-TUNING JOB SETUP")
        logger.info("=" * 70)
        
        # Validate setup
        validate_setup()
        
        # Setup environment
        setup_environment()
        
        # Create estimator
        estimator = create_estimator()
        
        # Save configuration
        save_job_config(estimator)
        
        # Print summary
        logger.info("=" * 70)
        logger.info("TRAINING JOB SUMMARY - VISION MODEL")
        logger.info("=" * 70)
        logger.info(f"Model: {estimator.hyperparameters()['model_name']}")
        logger.info(f"Dataset: {estimator.hyperparameters()['dataset_name']}")
        logger.info(f"Instruction: {estimator.hyperparameters()['instruction']}")
        logger.info(f"Instance: {INSTANCE_TYPE}")
        logger.info(f"Epochs: {estimator.hyperparameters().get('num_train_epochs', 'N/A')}")
        logger.info(f"Batch size: {estimator.hyperparameters()['per_device_train_batch_size']}")
        logger.info(f"Learning rate: {estimator.hyperparameters()['learning_rate']}")
        logger.info(f"LoRA r: {estimator.hyperparameters()['lora_r']}")
        logger.info(f"Source directory: {SOURCE_DIR}")
        logger.info(f"Entry point: train_vision.py")
        logger.info("=" * 70)
        
        # Start training
        logger.info("Starting training job...")
        estimator.fit(wait=False)  # Non-blocking call
        
        logger.info("=" * 70)
        logger.info(f"✓ Training job started: {estimator.latest_training_job.name}")
        logger.info("✓ You can monitor the job in the AWS SageMaker console")
        logger.info(f"✓ Config saved to: job_config_vision.json")
        logger.info("=" * 70)
        
        # Optionally wait for completion
        # estimator.fit(wait=True)
        
    except Exception as e:
        logger.error(f"Failed to start training job: {e}")
        raise

if __name__ == "__main__":
    main()
