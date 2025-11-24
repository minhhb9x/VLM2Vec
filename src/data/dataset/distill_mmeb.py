from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os
from copy import deepcopy
from datasets import Features, Value, Sequence

from torch.jit import isinstance
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils.basic_utils import print_master, print_rank
from torch.utils.data import Dataset

DISTILL_FEATURES = Features(**{
    "query_text": Value(dtype="string"),
    "pos_text": Value(dtype="string"),
    "neg_text": Value(dtype="string"),

    "student_query_text": Value(dtype="string"),
    "query_image": {
        "paths": Sequence(Value(dtype="string")),
        "bytes": Sequence(Value(dtype="binary")),
        "resolutions": Sequence(Sequence(Value(dtype="int32"), length=2)),
    },
    "student_pos_text": Value(dtype="string"),
    "pos_image": {
        "paths": Sequence(Value(dtype="string")),
        "bytes": Sequence(Value(dtype="binary")),
        "resolutions": Sequence(Sequence(Value(dtype="int32"), length=2)),
    },
    "student_neg_text": Value(dtype="string"),
    "neg_image": {
        "paths": Sequence(Value(dtype="string")),
        "bytes": Sequence(Value(dtype="binary")),
        "resolutions": Sequence(Sequence(Value(dtype="int32"), length=2)),
    },

    "teacher_query_text": Value(dtype="string"),
    "teacher_pos_text": Value(dtype="string"),
    "teacher_neg_text": Value(dtype="string"),

    "global_dataset_name": Value(dtype="string"),
})

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_dir = kwargs['image_dir']
    student_model_backbone = kwargs['student_model_backbone']
    teacher_model_backbone = kwargs['teacher_model_backbone']
    image_resolution = kwargs['image_resolution']

    def get_text(qry_text, pos_text, neg_text, model_backbone):
        if not qry_text and not pos_text:
            return '', '', ''
        if model_backbone == PHI3V:
            return qry_text, pos_text, neg_text
        qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else ''
        return qry_text, pos_text, neg_text

    batch_size = len(batch_dict['qry'])
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []    
    student_query_texts, student_pos_texts, student_neg_texts = [], [], []
    teacher_query_texts, teacher_pos_texts, teacher_neg_texts = [], [], []
    for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path in \
        zip(batch_dict['qry'], batch_dict['qry_image_path'],
            batch_dict['pos_text'], batch_dict['pos_image_path'],
            batch_dict.get('neg_text', [''] * batch_size), batch_dict.get('neg_image_path', [None] * batch_size)):
        if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
            print("empty inputs")
            continue

        student_qry_text, student_pos_text, student_neg_text = deepcopy(qry_text), deepcopy(pos_text), deepcopy(neg_text)
        teacher_qry_text, teacher_pos_text, teacher_neg_text = deepcopy(qry_text), deepcopy(pos_text), deepcopy(neg_text)
        
        student_qry_text, student_pos_text, student_neg_text = get_text(student_qry_text, student_pos_text, student_neg_text, student_model_backbone)
        teacher_qry_text, teacher_pos_text, teacher_neg_text = get_text(teacher_qry_text, teacher_pos_text, teacher_neg_text, teacher_model_backbone)

        query_texts.append(qry_text)
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)

        student_query_texts.append(student_qry_text)
        student_pos_texts.append(student_pos_text)
        student_neg_texts.append(student_neg_text)
        teacher_query_texts.append(teacher_qry_text)
        teacher_pos_texts.append(teacher_pos_text)
        teacher_neg_texts.append(teacher_neg_text)
        
        # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
        qry_image = {"bytes": [None], "paths": [os.path.join(image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        pos_image = {"bytes": [None], "paths": [os.path.join(image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        neg_image = {"bytes": [None], "paths": [os.path.join(image_dir, neg_image_path) if neg_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        

        
        query_images.append(qry_image)
        pos_images.append(pos_image)
        neg_images.append(neg_image)

    if len(student_query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "pos_text": pos_texts, "neg_text": neg_texts,
            "student_query_text": student_query_texts, "query_image": query_images,
            "student_pos_text": student_pos_texts, "pos_image": pos_images,
            "student_neg_text": student_neg_texts, "neg_image": neg_images,
            "teacher_query_text": teacher_query_texts, "teacher_pos_text": teacher_pos_texts,
            "teacher_neg_text": teacher_neg_texts}


DATASET_PARSER_NAME = "distill_mmeb"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_distill_mmeb_dataset(student_args, teacher_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "original")
    dataset = load_dataset(dataset_name, subset_name, split=f"{dataset_split}")
    column_names = dataset.column_names
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards

    kwargs['student_model_backbone'] = student_args.model_backbone
    kwargs['teacher_model_backbone'] = teacher_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    remove_columns = ['qry', 'qry_image_path', 'pos_image_path']
    if 'neg_image_path' in column_names:
        remove_columns.append('neg_image_path')
    dataset = dataset.map(lambda x:
                          data_prepare(x, **kwargs), batched=True, batch_size=2048,
                          remove_columns=remove_columns, drop_last_batch=False)
    # dataset = dataset._resolve_features()
    # features = _infer_features_from_batch(dataset._head()) # not working: {ArrowInvalid}ArrowInvalid('Could not convert <PIL.Image.Image image mode=RGB size=128x128 at 0x7F7C794E9BD0> with type Image: did not recognize Python value type when inferring an Arrow data type')
    dataset = dataset.cast(DISTILL_FEATURES)
    print_master(f"Loaded {DATASET_PARSER_NAME}/{subset_name} dataset with {num_rows} samples")

    # num_rows in iterable_dataset is overridden, set it here for printing dataset stats
    setattr(dataset, 'num_rows', num_rows)

    return dataset