import logging
import os.path
import sys

# PROJECT_ROOT = "/workspace/VLM2Vec"
# sys.path.append(PROJECT_ROOT)

from typing import Dict, Union
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
    def __init__(self,
            student: MMEBModel,
            teacher: MMEBModel
            ):
        super(DistillationModel, self).__init__()
        self.student = student
        self.teacher = teacher
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
    
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def build(cls, student_args: ModelArguments, teacher_args: Union[TeacherArguments, ModelArguments], data_args: DataArguments, training_args: TrainingArguments):
        teacher_args = map_teacher_to_model_args(teacher_args)
        if training_args.resume_from == 'auto':
            resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
            if resume_checkpoint_dir:
                logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
        elif training_args.resume_from.isdigit():
            resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
            if os.path.exists(resume_checkpoint_dir):
                logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
        else:
            resume_checkpoint_dir = None
            logger.info("No checkpoint found. Starting fresh training.")

        resume_checkpoint_dir = resume_checkpoint_dir
        logger.info(f"Set resume_checkpoint_dir to: {resume_checkpoint_dir}")

        # build student and teacher models
        student = MMEBModel.build(student_args)
        student_model_backbone = get_backbone_name(hf_config=student.config)
        setattr(training_args, 'student_model_backbone', student_model_backbone)
        setattr(student_args, 'model_backbone', student_model_backbone)
        student_processor = load_processor(student_args, data_args)
        setattr(student, 'processor', student_processor)


        hf_config = AutoConfig.from_pretrained(teacher_args.model_name, trust_remote_code=True)
        if not getattr(teacher_args, "model_backbone", None):
            teacher_model_backbone = get_backbone_name(hf_config=hf_config, model_type=teacher_args.model_type)
            setattr(teacher_args, 'model_backbone', teacher_model_backbone)
        
        teacher_model_backbone = teacher_args.model_backbone
        setattr(training_args, 'teacher_model_backbone', teacher_model_backbone)
        teacher_processor = load_processor(teacher_args, data_args)
        teacher = MMEBModel.load(teacher_args, is_trainable=False, processor=teacher_processor)
        setattr(teacher, 'processor', teacher_processor)

        print_rank(f'student_model_backbone: {student_model_backbone}')
        print_rank(f'teacher_model_backbone: {teacher_model_backbone}')
        print_rank(f'type of student processor: {type(student_processor)}')
        print_rank(f'type of teacher processor: {type(teacher_processor)}')
        print_rank(f'type of student model: {type(student.encoder)}')
        print_rank(f'type of teacher model: {type(teacher.encoder)}')

        model = cls(student=student, teacher=teacher)
        return {
            "model": model,
            "student_processor": student_processor,
            "teacher_processor": teacher_processor,
            "student_args": student_args,
            "teacher_args": teacher_args,
            "data_args": data_args,
            "training_args": training_args,
        }

    def save(self, output_dir: str):
        self.student.encoder.save_pretrained(output_dir)

    @classmethod
    def load(cls, 
             student_args: ModelArguments, 
             teacher_args: Union[TeacherArguments, ModelArguments], 
             data_args: DataArguments, 
             training_args: TrainingArguments,
             is_trainable: bool = True):
        teacher_args = map_teacher_to_model_args(teacher_args)
        model_info = cls.build(student_args, teacher_args, data_args, training_args)
        student_processor = model_info['student_processor']
        model = model_info['model']
        del model.student
        torch.cuda.empty_cache()
        student = MMEBModel.load(student_args, 
                                 is_trainable=is_trainable, 
                                 processor=student_processor)
        setattr(model, 'student', student)
        return model_info
    
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def forward(self, **inputs):
        student_input = inputs['student_inputs']
        teacher_input = inputs['teacher_inputs']

        stu_qry_reps = self.student.encode_input(student_input[0]) if student_input[0] else None
        stu_tgt_reps = self.student.encode_input(student_input[1]) if student_input[1] else None

        tea_qry_reps = self.teacher.encode_input(teacher_input[0]) if teacher_input[0] else None
        tea_tgt_reps = self.teacher.encode_input(teacher_input[1]) if teacher_input[1] else None

        return {
            'student_query_reps': stu_qry_reps,
            'student_target_reps': stu_tgt_reps,
            'teacher_query_reps': tea_qry_reps,
            'teacher_target_reps': tea_tgt_reps,
        }


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TeacherArguments, DataArguments, TrainingArguments))
    model_args, teacher_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    teacher_args: TeacherArguments
    data_args: DataArguments
    training_args: TrainingArguments

    model_info = DistillationModel.build(model_args, teacher_args, data_args, training_args)


  

    