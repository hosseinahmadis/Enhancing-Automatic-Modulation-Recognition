import torch
import torch.nn.functional as F
import math

def contrastive_loss(p1, p2, temperature=0.07):
    batch_size = p1.size(0)
    representations = torch.cat([p1, p2], dim=0)
    sim_matrix = torch.matmul(representations, representations.T) / temperature
    mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
    labels = torch.cat([
        torch.arange(batch_size) + batch_size,
        torch.arange(batch_size)
    ], dim=0).to(sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)

def compute_total_loss(
    logits, x_rec, source,source_aug, has_labeled, targets,
    projection1, projection2,
    criterion,
    weights,
    temperature=0.07
):
    # ----------------- Classification (Supervised + Pseudo-Labels) -----------------
    loss_classification = compute_classification_loss_with_pseudo_labels(
        logits=logits,
        targets=targets,
        has_labeled=has_labeled,
        confidence_threshold=weights.get("selftraining_confidence_threshold", 0.8),
        criterion=criterion,
        pseudo_label_weight=weights.get("pseudo_label_weight", 0.5)
    )

    # ----------------- Reconstruction -----------------
    loss_reconstruction = F.mse_loss(x_rec, source, reduction='none').sum() / x_rec.shape[0]

    # ----------------- Contrastive -----------------
    loss_contrastive = contrastive_loss(projection1, projection2, temperature=temperature)

    # ----------------- Total Weighted Loss -----------------
    loss = (
        weights['classification'] * loss_classification +
        weights['reconstruction'] * loss_reconstruction +
        weights['contrastive'] * loss_contrastive
    )

    if not math.isfinite(loss.item()):
        raise ValueError(f"Loss is {loss.item()}, stopping training")

    return loss, loss_classification, loss_reconstruction, loss_contrastive


def compute_classification_loss_with_pseudo_labels(
    logits,
    targets,
    has_labeled,
    confidence_threshold,
    criterion,
    pseudo_label_weight=0.5
):
    device = logits.device

    # --- Labeled classification loss ---
    if has_labeled.sum() > 0:
        loss_labeled = criterion(logits[has_labeled], targets[has_labeled])
    else:
        loss_labeled = torch.tensor(0.0, device=device)

    # --- Pseudo-labeling for unlabeled samples ---
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        confident_mask = (max_probs >= confidence_threshold) & (~has_labeled)

    if confident_mask.sum() > 0:
        loss_pseudo = criterion(logits[confident_mask], pseudo_labels[confident_mask])
    else:
        loss_pseudo = torch.tensor(0.0, device=device)

    # Combine both losses
    total_loss = loss_labeled + pseudo_label_weight * loss_pseudo

    return total_loss
