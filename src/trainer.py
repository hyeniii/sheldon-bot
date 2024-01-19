import logging
import os
from typing import List
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

import src.helper as h
from src.evaluate import evaluate

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Train the model"""
    # Set up TensorBoard writer if running on the main GPU or 
    # in a non-distributed setup
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    # Calculate the training batch size, considering the number of GPUs
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # Function to properly collate batch data, 
    # handling padding based on tokenizer settings
    def collate(examples: List[torch.tensor]):
        if not tokenizer._pad_token:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # Set up data sampler: 
    # distributed sampler for distributed training, else random sampler
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else \
                    DistributedSampler(train_dataset)
    
    # DataLoader to batch and load the training data
    train_dataloader = DataLoader(
        train_dataset, 
        sampler = train_sampler, 
        batch_size = args.train_batch_size,
        collate_fn = collate,
        drop_last=True
    )

    # step = one gradient update on a batch
    # epoch = one complete pass through entire data (all batches)
    # num. parameter updates per epoch = num. batches / accumulate every n batches
    # Calculate the number of parameter updates per epoch
    param_updates_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps

    if args.max_steps > 0:
        # total number of training steps
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // param_updates_per_epoch + 1
    else:
        t_total = param_updates_per_epoch * args.num_train_epochs
    
    # Adjust model for distributed training if needed
    model = model.module if hasattr(model, "module") else model
    # Update model's token embeddings to match tokenizer's vocabulary size
    model.resize_token_embeddings(len(tokenizer))


    # Prepare optimizer and learning rate scheduler
    # Define groups of model parameters with and without weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_params, lr= args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Load optimizer and scheduler states if resuming training from a checkpoint
    optimizer_path = os.path.join(args.model_name_or_path, "optimizer.pt")
    scheduler_path = os.path.join(args.model_name_or_path, "scheduler.pt")
    if (
        args.model_name_or_path 
        and os.path.isfile(optimizer_path) 
        and os.path.isfile(scheduler_path)
    ):
        # load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(optimizer_path))
        scheduler.load_state_dict(torch.load(scheduler_path))

    # Initialize mixed precision training using NVIDIA's apex if fp16 flag is set
    if args.fp16:
        try:
            from apex import amp 
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    # If multiple GPUs are available, use DataParallel for parallel training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # For distributed training, wrap the model in DistributedDataParallel
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids = [args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    # Initialize training variables
    global_step = 0
    epochs_trained = 0
    steps_trained_in_cur_epoch = 0

    # Check if resuming training from a saved checkpoint
    if args.model_name_or_path or os.path.exists(args.model_name_or_path):
        # Extract checkpoint information for resuming training
        try:
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // param_updates_per_epoch
            steps_trained_in_cur_epoch = global_step % param_updates_per_epoch
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_cur_epoch)
        except ValueError:
            logger.info("   Starting fine-tuning")
    
    train_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = tqdm(
        range(epochs_trained, int(args.num_train_epochs)), 
        desc="Epoch", 
        disable=args.local_rank not in [-1,0]
    )
    h.set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            
            # Skip already trained steps if resuming training
            if steps_trained_in_cur_epoch:
                steps_trained_in_cur_epoch -= 1
                continue
            
            inputs, labels = (batch, batch)
            if inputs.shape[1] > 1024: continue 
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            # Set model to training mode and perform forward pass
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            # Average loss across multiple GPUs
            if args.n_gpu > 1:
                loss = loss.mean()
            # Divide loss by accumulation steps for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Backward pass: compute gradient of the loss with respect to model parameters
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_loss += loss.item()
            # Update model parameters and learning rate every `gradient_accumulation_steps`
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients to prevent the "exploding gradient" problem
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()  # Apply gradients to update model parameters
                scheduler.step()  # Update learning rate
                model.zero_grad()  # Reset gradients
                global_step += 1  # Increment global step

                # Logging metrics and saving model checkpoints
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics if on the main process
                    # Optionally evaluate model and log results
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    # Log learning rate and training loss
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (train_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = train_loss

                # Save model checkpoint periodically
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    h._rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            # Break from the loop if the maximum number of steps is reached
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        # Check again for max_steps to break from outer loop
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    # Close the TensorBoard writer if it was opened
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # Return global step and average training loss
    return global_step, train_loss / global_step