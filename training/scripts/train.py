#!/usr/bin/env python3
"""
Train Vedic Astrology LLM using LoRA/QLoRA fine-tuning
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


@dataclass
class ModelArguments:
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Path to pretrained model"}
    )
    use_8bit: bool = field(
        default=True,
        metadata={"help": "Load model in 8-bit mode"}
    )


@dataclass
class DataArguments:
    train_file: str = field(
        metadata={"help": "Path to training data"}
    )
    validation_file: str = field(
        metadata={"help": "Path to validation data"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules"}
    )


class VedicAstroTrainer:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        lora_args: LoraArguments,
        training_args: TrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.training_args = training_args
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization"""
        print("=" * 60)
        print("Loading Model and Tokenizer")
        print("=" * 60)
        
        # Check for CUDA availability
        has_cuda = torch.cuda.is_available()
        print(f"✓ CUDA available: {has_cuda}")
        if has_cuda:
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Quantization config (only if CUDA available)
        bnb_config = None
        device_map = None
        
        if self.model_args.use_8bit and has_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            device_map = "auto"
            print("✓ Using 8-bit quantization")
        elif not has_cuda:
            print("⚠️  No CUDA detected - training on CPU (will be slow)")
            device_map = "cpu"
        else:
            device_map = "auto"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            trust_remote_code=True,
            padding_side="right",
            add_eos_token=True,
        )
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✓ Loaded tokenizer: {self.model_args.model_name}")
        
        # Load model with proper device handling
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.float16
        elif has_cuda:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        # Only use device_map if not using accelerate's auto device placement
        if device_map != "auto":
            model_kwargs["device_map"] = device_map
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name,
            **model_kwargs
        )
        
        print(f"✓ Loaded model: {self.model_args.model_name}")
        
        # Prepare for k-bit training (only if using quantization)
        if self.model_args.use_8bit and has_cuda:
            model = prepare_model_for_kbit_training(model)
            print("✓ Prepared model for k-bit training")
        
        return model, tokenizer
    
    def setup_lora(self, model):
        """Setup LoRA configuration"""
        print("\n" + "=" * 60)
        print("Setting up LoRA")
        print("=" * 60)
        
        # Freeze all base model parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        target_modules = self.lora_args.lora_target_modules.split(",")
        
        lora_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        
        # Enable gradient checkpointing if needed
        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ LoRA Config:")
        print(f"  - Rank (r): {self.lora_args.lora_r}")
        print(f"  - Alpha: {self.lora_args.lora_alpha}")
        print(f"  - Dropout: {self.lora_args.lora_dropout}")
        print(f"  - Target modules: {target_modules}")
        print(f"\n✓ Trainable parameters: {trainable_params:,} / {total_params:,}")
        print(f"  ({100 * trainable_params / total_params:.2f}%)")
        
        # Verify we have trainable parameters
        if trainable_params == 0:
            raise ValueError("No trainable parameters found! LoRA setup failed.")
        
        return model
    
    def load_datasets(self, tokenizer, max_train_samples=None, max_eval_samples=None):
        """Load and preprocess datasets"""
        print("\n" + "=" * 60)
        print("Loading Datasets")
        print("=" * 60)
        
        # Load datasets
        dataset = load_dataset(
            "json",
            data_files={
                "train": self.data_args.train_file,
                "validation": self.data_args.validation_file,
            }
        )
        
        # Limit samples if specified (for testing)
        if max_train_samples is not None and max_train_samples < len(dataset["train"]):
            dataset["train"] = dataset["train"].select(range(max_train_samples))
            print(f"⚠️  Limited to {max_train_samples} training samples for testing")
        
        if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
            dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
            print(f"⚠️  Limited to {max_eval_samples} validation samples for testing")
        
        print(f"✓ Train samples: {len(dataset['train'])}")
        print(f"✓ Validation samples: {len(dataset['validation'])}")
        
        # Tokenization function
        def tokenize_function(examples):
            # Tokenize the text field
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.data_args.max_seq_length,
                padding="max_length",
            )
            
            # Labels are the same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize datasets
        print("\n✓ Tokenizing datasets...")
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )
        
        return tokenized_datasets
    
    def train(self, max_train_samples=None, max_eval_samples=None):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60 + "\n")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Setup LoRA
        model = self.setup_lora(model)
        
        # Load datasets
        tokenized_datasets = self.load_datasets(tokenizer, max_train_samples, max_eval_samples)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )
        
        # Train
        print("\n" + "=" * 60)
        print("Training Started")
        print("=" * 60 + "\n")
        
        trainer.train()
        
        # Save final model
        print("\n" + "=" * 60)
        print("Saving Model")
        print("=" * 60)
        
        output_dir = Path(self.training_args.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model saved to {output_dir}")
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Vedic Astrology LLM")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create arguments
    model_args = ModelArguments(
        model_name=config["model"]["name"],
        use_8bit=config["model"].get("load_in_8bit", True),
    )
    
    data_args = DataArguments(
        train_file=config["data"]["train_file"],
        validation_file=config["data"]["validation_file"],
        max_seq_length=config["training"].get("max_seq_length", 2048),
    )
    
    lora_args = LoraArguments(
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        lora_target_modules=",".join(config["lora"]["target_modules"]),
    )
    
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        eval_strategy=config["training"].get("eval_strategy", config["training"].get("evaluation_strategy", "steps")),
        save_total_limit=config["training"]["save_total_limit"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        fp16=config["training"]["fp16"],
        optim=config["training"]["optim"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        max_grad_norm=config["training"]["max_grad_norm"],
        report_to=config["training"]["report_to"],
        logging_dir=config["training"]["logging_dir"],
    )
    
    # Get sample limits from config
    max_train_samples = config["data"].get("max_train_samples")
    max_eval_samples = config["data"].get("max_eval_samples")
    
    # Train
    trainer = VedicAstroTrainer(model_args, data_args, lora_args, training_args)
    trainer.train(max_train_samples=max_train_samples, max_eval_samples=max_eval_samples)


if __name__ == "__main__":
    main()