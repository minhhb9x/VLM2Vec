import logging
import os.path
import sys

from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import HfArgumentParser

from src.arguments import ModelArguments, DataArguments, TrainingArguments, TeacherArguments
from src.model.processor import load_processor, get_backbone_name
from src.utils.basic_utils import print_rank, print_master, find_latest_checkpoint
from src.model.model import MMEBModel

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

class DistillationModel(nn.Module):
    def __init__(self, student_args: ModelArguments, teacher_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        super(DistillationModel, self).__init__()
        self.student_args = student_args
        self.teacher_args = teacher_args
        self.data_args = data_args
        self.training_args = training_args
        self._load()
    
    def _load(self):

        if self.training_args.resume_from == 'auto':
            resume_checkpoint_dir = find_latest_checkpoint(self.training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
        elif self.training_args.resume_from.isdigit():
            resume_checkpoint_dir = os.path.join(self.training_args.output_dir, f'checkpoint-{self.training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
        else:
            resume_checkpoint_dir = None
            logger.info("No checkpoint found. Starting fresh training.")

        self.resume_checkpoint_dir = resume_checkpoint_dir
        logger.info(f"Set resume_checkpoint_dir to: {self.resume_checkpoint_dir}")


        self.student_processor = load_processor(self.student_args, self.data_args)
        self.teacher_processor = load_processor(self.teacher_args, self.data_args)

        self.student = MMEBModel.build(self.student_args)
        self.teacher = MMEBModel.load(self.teacher_args, is_trainable=False, processor=teacher_processor)

        student_model_backbone = get_backbone_name(hf_config=self.student.config)
        teacher_model_backbone = get_backbone_name(hf_config=self.teacher.config)

        setattr(self.student_args, 'model_backbone', student_model_backbone)
        setattr(self.teacher_args, 'model_backbone', teacher_model_backbone)
        setattr(self.training_args, 'model_backbone', student_model_backbone)

        print_rank(f'student_model_backbone: {student_model_backbone}')
        print_rank(f'teacher_model_backbone: {teacher_model_backbone}')

        student_processor = load_processor(self.student_args, self.data_args)
        setattr(self.student, 'processor', student_processor)
        teacher_processor = load_processor(self.teacher_args, self.data_args)
        setattr(self.teacher, 'processor', teacher_processor)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TeacherArguments, DataArguments, TrainingArguments))
    model_args, teacher_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    teacher_args: TeacherArguments
    data_args: DataArguments
    training_args: TrainingArguments

    distillation_model = DistillationModel(model_args, teacher_args, data_args, training_args)

    