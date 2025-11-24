import logging
import os.path
import sys
from typing import Any, Union, Mapping

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import sys
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, TeacherArguments, TrainingArguments
from src.utils.basic_utils import batch_to_device
from src.data.loader.mixed_dataset import init_mixed_distill_dataset
from src.distillation.model import DistillationModel
from src.distillation.trainer import DistillTrainer
from src.distillation.collator import DistillMultimodalDataCollator
from src.utils.basic_utils import print_rank, print_master, find_latest_checkpoint
from src.model.processor import load_processor, get_backbone_name


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, TeacherArguments, DataArguments, TrainingArguments))

    student_args, teacher_args, data_args, training_args = parser.parse_args_into_dataclasses()
    student_args: ModelArguments
    teacher_args: TeacherArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # DEBUG PRINTS for Distributed Setup
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")

    model_info = DistillationModel.build(student_args, teacher_args, data_args, training_args)

    distill_model = model_info['model']
    student_processor = model_info['student_processor']
    teacher_processor = model_info['teacher_processor']
    student_args = model_info['student_args']
    teacher_args = model_info['teacher_args']
    data_args = model_info['data_args']
    training_args = model_info['training_args']
    
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
        if data_args.data_basedir:
            for _, task_config in dataset_config.items():
                image_dir = task_config.get('image_dir')
                if image_dir and not os.path.isabs(image_dir):
                    task_config['image_dir'] = os.path.join(data_args.data_basedir, image_dir)
        train_dataset = init_mixed_distill_dataset(dataset_config, student_args, teacher_args, data_args, training_args)
    train_collator = DistillMultimodalDataCollator(
        student_processor=student_processor,
        teacher_processor=teacher_processor,
        student_model_args=student_args,
        data_args=data_args,
        training_args=training_args,
        teacher_model_args=teacher_args,
        batch_size=training_args.per_device_train_batch_size,)
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=train_collator,
        num_workers=training_args.dataloader_num_workers,
    )
    distill_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    batch = next(iter(dataloader))
    
    student_qry, student_pos = batch['student']
    teacher_qry, teacher_pos = batch['teacher']
    student_qry = batch_to_device(student_qry, distill_model.student.device)
    student_pos = batch_to_device(student_pos, distill_model.student.device)
    teacher_qry = batch_to_device(teacher_qry, distill_model.teacher.device)
    teacher_pos = batch_to_device(teacher_pos, distill_model.teacher.device)
    batch['student'] = (student_qry, student_pos)
    batch['teacher'] = (teacher_qry, teacher_pos)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = distill_model(batch)

    print(output['student_query_reps'].shape)
    print(output['teacher_query_reps'].shape)
    
if __name__ == "__main__":
    main()