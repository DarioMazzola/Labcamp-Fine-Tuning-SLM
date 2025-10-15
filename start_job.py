import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
import json
from datetime import datetime
import logging
import getpass

username = getpass.getuser()  # Automatically gets the current logged-in user

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (loaded from environment when available)
REGION = os.getenv("AWS_REGION", "us-east-1")
ROLE = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::<ACCOUNT-ID>:role/<ROLE-NAME>")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.p3.2xlarge")
VOLUME_SIZE = int(os.getenv("VOLUME_SIZE", "150"))
IMAGE_NAME = os.getenv("ECR_IMAGE_NAME", "docker-image-name")  # e.g., unsloth/unsloth
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET", "")
SOURCE_DIR = os.getenv("SOURCE_DIR", "./training_file")  # Path to training code directory

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
    return f"gemma-function-calling-{timestamp}"

def get_hyperparameters():
    """Get hyperparameters with improved configuration"""
    hyperparameters = {
        # Model and data
        "model_name": "unsloth/gemma-3-4b-it",
        "dataset_name": "rewoo/planner_instruction_tuning_2k",
        "use_hub_dataset": "true",
        #"max_dataset_size": "10",
        
        # Training parameters
        "num_train_epochs": "3",
        "per_device_train_batch_size": "1",
        "gradient_accumulation_steps": "8",
        "learning_rate": "1e-4", 
        "max_seq_length": "4096", 
        
        # Evaluation and saving
        "eval_steps": "200",
        "logging_steps": "20",
        "save_steps": "400",
        "save_strategy": "steps",
        
        # Optimization
        "packing": "true",
        "bf16": "true",
        "load_in_4bit": "true",
        "load_in_8bit": "false",
        "gradient_checkpointing": "true",
        
        # LoRA parameters
        "lora_r": "32",
        "lora_alpha": "64",
        "lora_dropout": "0.05",
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
            raise FileNotFoundError(f"Create {SOURCE_DIR} and add your train.py there")
        
        train_py = os.path.join(SOURCE_DIR, "train.py")
        if not os.path.exists(train_py):
            logger.error(f"train.py not found in {SOURCE_DIR}")
            raise FileNotFoundError(f"Add train.py to {SOURCE_DIR}")
        
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
    
    wandb_token = os.getenv("WANDB_API_KEY")
    if wandb_token:
        logger.info("WandB token found in environment")

def create_estimator():
    """Create and configure the SageMaker estimator"""
    boto3_session = boto3.session.Session(region_name=REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto3_session)
    
    hyperparameters = get_hyperparameters()
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
        max_run=24 * 60 * 60,
        hyperparameters=hyperparameters,
        environment=environment,
        sagemaker_session=sagemaker_session,
        base_job_name=username,
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {"Name": "train_loss", "Regex": "train_loss: ([0-9\\.]+)"},
            {"Name": "eval_loss", "Regex": "eval_loss: ([0-9\\.]+)"},
            {"Name": "learning_rate", "Regex": "learning_rate: ([0-9\\.]+)"},
        ],
        tags=[
            {"Key": "Project", "Value": "FunctionCalling"},
            {"Key": "Model", "Value": "Gemma3-4B"},
            {"Key": "Environment", "Value": "Development"},
        ],
        # NUOVO: Specifico il source directory e l'entry point
        source_dir=SOURCE_DIR,
        entry_point="train.py",  # File da eseguire dentro SOURCE_DIR
        output_path=S3_OUTPUT_BUCKET,
        code_location=SOURCE_DIR,
    )
    
    return estimator

def save_job_config(estimator, output_file="training_file/job_config.json"):
    """Save job configuration for reference"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "instance_type": INSTANCE_TYPE,
        "volume_size": VOLUME_SIZE,
        "hyperparameters": estimator.hyperparameters(),
        "role": ROLE,
        "image_uri": estimator.image_uri,
        "source_dir": SOURCE_DIR,
    }
    
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Job configuration saved to {output_file}")

def main():
    """Main execution function"""
    try:
        logger.info("Starting SageMaker training job setup...")
        
        # Validate setup
        validate_setup()
        
        # Setup environment
        setup_environment()
        
        # Create estimator
        estimator = create_estimator()
        
        # Save configuration
        save_job_config(estimator)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TRAINING JOB SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model: {estimator.hyperparameters()['model_name']}")
        logger.info(f"Dataset: {estimator.hyperparameters().get('dataset_name')}")
        logger.info(f"Instance: {INSTANCE_TYPE}")
        logger.info(f"Epochs: {estimator.hyperparameters().get('num_train_epochs', 'N/A')}")
        logger.info(f"Batch size: {estimator.hyperparameters().get('per_device_train_batch_size')}")
        logger.info(f"Learning rate: {estimator.hyperparameters().get('learning_rate')}")
        logger.info(f"Source directory: {SOURCE_DIR}")
        logger.info("=" * 50)
        
        # Start training
        logger.info("Starting training job...")
        estimator.fit(wait=False)  # Non-blocking call
        
        logger.info(f"Training job started: {estimator.latest_training_job.name}")
        logger.info("You can monitor the job in the AWS SageMaker console")
        
        # Optionally wait for completion
        # estimator.fit(wait=True)
        
    except Exception as e:
        logger.error(f"Failed to start training job: {e}")
        raise

if __name__ == "__main__":
    main()