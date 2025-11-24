from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F

class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, output: dict, reduction: str = 'mean') -> Tensor:
        student_x = output['student_query_reps']  # shape=[bsz, hdim]
        student_y = output['student_target_reps']  # shape=[bsz, hdim]
        # teacher_x = output['teacher_query_reps']  # shape=[bsz, h1dim]
        # teacher_y = output['teacher_target_reps']  # shape=[bsz, h1dim]

        logits_student = torch.matmul(student_x, student_y.transpose(0, 1))  # shape=[bsz, bsz]
        # logits_teacher = torch.matmul(teacher_x, teacher_y.transpose(0, 1))  # shape=[bsz, bsz]

        target = torch.arange(0, student_x.size(0), device=student_x.device, dtype=torch.long)
        loss_student = F.cross_entropy(logits_student / self.temperature, target, reduction=reduction)
        return loss_student
    

class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, output: dict, **kwargs):
        student_x = self.gather_tensor(output['student_query_reps'])
        student_y = self.gather_tensor(output['student_target_reps'])
        output_dist = {
            'student_query_reps': student_x,
            'student_target_reps': student_y,
        }
        loss = super().__call__(output_dist, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)