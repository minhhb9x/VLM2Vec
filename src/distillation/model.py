import logging
import os.path
import sys

PROJECT_ROOT = "/workspace/VLM2Vec"
sys.path.append(PROJECT_ROOT)

from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import HfArgumentParser, AutoConfig

from src.arguments import ModelArguments, DataArguments, TrainingArguments, TeacherArguments, map_teacher_to_model_args
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
        self.teacher_args = map_teacher_to_model_args(teacher_args)
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

        self.student = MMEBModel.build(self.student_args)
        student_model_backbone = get_backbone_name(hf_config=self.student.config)
        setattr(self.training_args, 'student_model_backbone', student_model_backbone)
        setattr(self.student_args, 'model_backbone', student_model_backbone)
        self.student_processor = load_processor(self.student_args, self.data_args)
        setattr(self.student, 'processor', self.student_processor)


        hf_config = AutoConfig.from_pretrained(self.teacher_args.model_name, trust_remote_code=True)
        if not getattr(self.teacher_args, "model_backbone", None):
            teacher_model_backbone = get_backbone_name(hf_config=hf_config, model_type=self.teacher_args.model_type)
            setattr(self.teacher_args, 'model_backbone', teacher_model_backbone)
        
        teacher_model_backbone = self.teacher_args.model_backbone
        setattr(self.training_args, 'teacher_model_backbone', teacher_model_backbone)
        self.teacher_processor = load_processor(self.teacher_args, self.data_args)
        self.teacher = MMEBModel.load(self.teacher_args, is_trainable=False, processor=self.teacher_processor)
        setattr(self.teacher, 'processor', self.teacher_processor)

        print_rank(f'student_model_backbone: {student_model_backbone}')
        print_rank(f'teacher_model_backbone: {teacher_model_backbone}')
        print_rank(f'type of student processor: {type(self.student_processor)}')
        print_rank(f'type of teacher processor: {type(self.teacher_processor)}')
        print_rank(f'type of student model: {type(self.student.encoder)}')
        print_rank(f'type of teacher model: {type(self.teacher.encoder)}')


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TeacherArguments, DataArguments, TrainingArguments))
    model_args, teacher_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    teacher_args: TeacherArguments
    data_args: DataArguments
    training_args: TrainingArguments

    distillation_model = DistillationModel(model_args, teacher_args, data_args, training_args)

    