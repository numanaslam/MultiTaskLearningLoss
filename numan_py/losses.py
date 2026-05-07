# losses.py
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.5, gamma=2.0, eps=1e-7):
    probs = F.softmax(logits, dim=1)
    probs = probs.clamp(eps, 1-eps)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
    alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
    return -(alpha_t * (1 - pt)**gamma * torch.log(pt)).mean()

def cam_cosine_loss(stud_cam, target_cam):
    stud = stud_cam.view(stud_cam.size(0), -1)
    target = target_cam.view(target_cam.size(0), -1)
    cos_sim = F.cosine_similarity(stud, target, dim=1)
    return (1 - cos_sim).clamp(0, 2).mean()

def tversky_loss(pred_mask, target_mask, alpha=0.7, beta=0.3, eps=1e-6):
    tp = (pred_mask * target_mask).sum(dim=(1,2,3))
    fp = (pred_mask * (1 - target_mask)).sum(dim=(1,2,3))
    fn = ((1 - pred_mask) * target_mask).sum(dim=(1,2,3))
    return (1 - tp / (tp + alpha*fp + beta*fn + eps)).mean()

def anatomical_loss(stud_cam, lung_mask, reward_weight=0.75, scale_factor=1000):
    cam_norm = stud_cam / (stud_cam.amax(dim=(1,2,3), keepdim=True) + 1e-8)
    penalty = (cam_norm * (1 - lung_mask)).mean(dim=(1,2,3))
    reward = (cam_norm * lung_mask).mean(dim=(1,2,3))
    loss = (penalty - reward_weight * reward).clamp(min=0) * scale_factor
    return loss.mean()