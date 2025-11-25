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


    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        logger.info(f"Loading checkpoint from {resume_from_checkpoint}")
        
        if not isinstance(self.model, DistillationModel):
            self.student_args.checkpoint_path = resume_from_checkpoint
            student = MMEBModel.load(self.student_args)
            teacher = MMEBModel.load(self.teacher_args, is_trainable=False)
            self.model = DistillationModel(student=student, teacher=teacher)
        
        full_distill_model = self.model 
        
        self.model = full_distill_model.student 
        
        super()._load_from_checkpoint(resume_from_checkpoint, model=self.model)
        
        self.model = full_distill_model
        
        self.model_wrapped = self.model

        adapter_path = os.path.join(resume_from_checkpoint, "distill_adapters.bin")
        if os.path.exists(adapter_path):
            logger.info(f"Loading adapters from {adapter_path}")
            adapter_state = torch.load(adapter_path, map_location="cpu")
            
            self.model.student_adapter.load_state_dict(adapter_state['student_adapter'])
            self.model.teacher_adapter.load_state_dict(adapter_state['teacher_adapter'])
        else:
            logger.warning(f"No adapter weights found at {adapter_path}, using random init.")

    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print_master(f"Saving student model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        self.model.save(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        self.model.student.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))