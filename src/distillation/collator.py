from typing import Optional, Union
from copy import deepcopy

import logging
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments, TeacherArguments, TrainingArguments, map_teacher_to_model_args

from src.data.collator.train_collator import MultimodalDataCollator
from src.utils.basic_utils import print_rank, print_master

logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

class DistillMultimodalDataCollator:
    def __init__(self,
                student_processor: ProcessorMixin,
                teacher_processor: ProcessorMixin,
                student_model_args: ModelArguments,
                data_args: DataArguments,
                training_args: TrainingArguments,
                teacher_model_args: Union[ModelArguments, TeacherArguments],
                batch_size: Optional[int] = None  # used to verify if a batch has invalid data
                 ):
        student_training_args = deepcopy(training_args)
        teacher_training_args = deepcopy(training_args)

        setattr(student_training_args, 'model_backbone', student_training_args.student_model_backbone)
        setattr(teacher_training_args, 'model_backbone', teacher_training_args.teacher_model_backbone)

        self.student_collator = MultimodalDataCollator(
            processor=student_processor,
            model_args=student_model_args,
            data_args=data_args,
            training_args=student_training_args,
            batch_size=batch_size,
        )
        teacher_model_args = map_teacher_to_model_args(teacher_model_args)
        self.teacher_collator = MultimodalDataCollator(
            processor=teacher_processor,
            model_args=teacher_model_args,
            data_args=data_args,
            training_args=teacher_training_args,
            batch_size=batch_size,
        )

    def __call__(self, examples):
        student_examples = [
            {
                "query_text": ex["student_query_text"],
                "pos_text": ex["student_pos_text"],
                "neg_text": ex["student_neg_text"],
                "query_image": ex["query_image"],
                "pos_image": ex["pos_image"],
                "neg_image": ex["neg_image"],
                "global_dataset_name": ex["global_dataset_name"],
            }
            for ex in examples
        ]

        teacher_examples = [
            {
                "query_text": ex["teacher_query_text"],
                "pos_text": ex["teacher_pos_text"],
                "neg_text": ex["teacher_neg_text"],
                "query_image": ex["query_image"],
                "pos_image": ex["pos_image"],
                "neg_image": ex["neg_image"],
                "global_dataset_name": ex["global_dataset_name"],
            }
            for ex in examples
        ]

        student_batch = self.student_collator(student_examples)
        teacher_batch = self.teacher_collator(teacher_examples)

        return {
            'student': student_batch,
            'teacher': teacher_batch,
        }