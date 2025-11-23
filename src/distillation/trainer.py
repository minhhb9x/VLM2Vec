import os
import contextlib
import functools
import shutil
import sys
import time
from datetime import timedelta
from typing import Optional, Union

import torch
import torch.distributed as dist

from transformers.trainer import Trainer, TRAINING_ARGS_NAME, TRAINER_STATE_NAME
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_apex_available,
    is_torch_xla_available,
    logging, is_sagemaker_mp_enabled,
    CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME
)

from src.arguments import ModelArguments, TeacherArguments, map_teacher_to_model_args
from src.trainer import MMEBTrainer
from src.model.model import MMEBModel

from src.distillation.model import DistillationModel

from src.utils.basic_utils import batch_to_device
from src.utils.basic_utils import print_master

logger = logging.get_logger(__name__)

class DistillTrainer(MMEBTrainer):
    def __init__(self, 
                 student_args: ModelArguments,
                 teacher_args: Union[TeacherArguments, ModelArguments],
                 student_processing_class,
                 teacher_processing_class,
                 *args, **kwargs):
        self.student_processing_class = student_processing_class
        self.teacher_processing_class = teacher_processing_class
        self.student_args = student_args
        self.teacher_args = map_teacher_to_model_args(teacher_args)
        super(DistillTrainer).__init__(
            processing_class=student_processing_class, 
            *args, 
            **kwargs)


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print_master(f"Saving student model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.student.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.student.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.student.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))


    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.student_args.checkpoint_path = resume_from_checkpoint
        logger.info(f"Loading student checkpoint from {resume_from_checkpoint}")
        student = MMEBModel.load(self.student_args)
        teacher = MMEBModel.load(self.teacher_args, is_trainable=False)
        self.model = DistillationModel(student=student, teacher=teacher)
        self.model_wrapped = self.model