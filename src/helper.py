import glob
import logging
import os
import random
import re
import shutil

import numpy as np
import torch

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
)
from src.dataset import ConversationDataset

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def load_and_cache_examples(args, tokenizer, df_train, df_val, evaluate=False):
    return ConversationDataset(tokenizer, args, df_val if evaluate else df_train)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
    checkpoint_sorted = sorted(ordering_and_checkpoint_path)
    checkpoint_sorted = [checkpoint[1] for checkpoint in checkpoint_sorted]
    return checkpoint_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    if not args.save_total_limit:
        return 
    if args.save_total_limit <= 0:
        return
    checkpoint_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoint_sorted) <= args.save_total_limit:
        return
    num_checkpoints_to_delete = max(0, len(checkpoint_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoint_sorted[:num_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)