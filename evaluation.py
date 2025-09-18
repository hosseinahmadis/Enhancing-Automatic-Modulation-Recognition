import torch
from losses import compute_total_loss
import os
import torch
import time
from dataset import RML2018_Dataset
from torch.utils.data import DataLoader
from utils import LogToFile, get_device,get_criterion
from losses import compute_total_loss
def evaluate_model(model, dataloader, criterion, args, device, logger=None):
    model.to(device)
    model.eval()

    weights = {
            'classification': args.w_classification,
            'reconstruction': args.w_reconstruction,
            'contrastive': args.w_contrastive,
            "pseudo_label_weight": args.pseudo_label_weight,
            "selftraining_confidence_threshold": args.selftraining_confidence_threshold
        }
    temperature = args.contrastive_temperature
    print_acc_per_class = args.print_acc_per_class
    current_classes = args.current_classes

    total_loss = 0
    total_loss_classification = 0
    total_loss_reconstruction = 0
    total_loss_contrastive = 0
    correct_preds = 0
    total_samples = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for batch in dataloader:
            source = batch['iq'].to(device)
            source_augment = batch['iq_transformed'].to(device)
            source_1 = batch['iq_rotated'].to(device)
            source_2 = batch['iq_flipped'].to(device)
            targets = batch['label'].to(device)
            # Force has_labeled = True for all in evaluation
            has_labeled = torch.ones_like(targets, dtype=torch.bool).to(device)

            output = model(source)
            output1 = model(source_1)
            output2 = model(source_2)

            logits = output.get("logits", None)
            x_rec = output.get("x_rec", None)
            projection1 = output1.get("average_patch_proj", None)
            projection2 = output2.get("average_patch_proj", None)

            loss, loss_cls, loss_rec, loss_con = compute_total_loss(
                logits=logits,
                x_rec=x_rec,
                source=source,
                source_aug=source_augment,  # Assuming source_1 is the augmented source
                has_labeled=has_labeled,
                targets=targets,
                projection1=projection1,
                projection2=projection2,
                criterion=criterion,
                weights=weights,
                temperature=temperature
            )

            total_loss += loss.item()
            total_loss_classification += loss_cls.item()
            total_loss_reconstruction += loss_rec
            total_loss_contrastive += loss_con.item()

            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            for label, pred in zip(targets, predicted):
                label = label.item()
                class_correct[label] = class_correct.get(label, 0) + (pred == label).item()
                class_total[label] = class_total.get(label, 0) + 1

    avg_loss = total_loss / len(dataloader)
    avg_cls = total_loss_classification / len(dataloader)
    avg_rec = total_loss_reconstruction / len(dataloader)
    avg_con = total_loss_contrastive / len(dataloader)
    accuracy = (correct_preds / total_samples) * 100

    msg = (
        f"\n📊 Validation Summary\n"
        f"  Accuracy:              {accuracy:.2f}%\n"
        f"  Avg Total Loss:        {avg_loss:.4f}\n"
        f"  Classification Loss:   {avg_cls:.4f}\n"
        f"  Reconstruction Loss:   {avg_rec:.4f}\n"
        f"  Contrastive Loss:      {avg_con:.4f}\n"
    )
    if logger:
        

        if print_acc_per_class:
            logger(msg)
            per_class_msg = "📌 Per-Class Accuracy:\n"
            for label in sorted(class_correct.keys()):
                acc = (class_correct[label] / class_total[label]) * 100
                per_class_msg += f"  {current_classes[label]:<15}: {acc:.2f}%\n"
            logger(per_class_msg)

    return {
        "avg_loss": avg_loss,
        "avg_cls": avg_cls,
        "avg_rec": avg_rec,
        "avg_con": avg_con,
        "accuracy": accuracy,
        "class_correct": class_correct,
        "class_total": class_total
    }



def test_model(model, run_number, test_loader, args, config_model):
    project_title = args.project_title
    result_path = args.result_path

    test_logger = LogToFile(
        filename=os.path.join(result_path, f"test_model_{project_title}_run{run_number}.txt"),
        enable=True
    )
    test_logger.clear_log_file()
    test_logger(f"🧪 Starting test evaluation...              Run {run_number}")

    test_logger("🧪 Test Configuration:")
    if hasattr(args, 'grid_search_vit'):
        test_logger(f"\n🧪 grid_search_vit settings:")
        for key, values in args.grid_search_vit.items():
            test_logger(f"  {key}: {values}")
    test_logger(f"Current Configuration:     {config_model}")
    test_logger(f" ------------------------------------------------------------------------------------------------------------------------------\n\n")

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_logger(f"📐 Model Parameters:")
    test_logger(f"  Total parameters:     {total_params:,}")
    test_logger(f"  Trainable parameters: {trainable_params:,}")

    device = get_device()
    model.to(device)
    model.eval()

    start_time = time.time()
    args.print_acc_per_class = True

    # Run evaluation
    eval_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=get_criterion(args.loss_function),
        args=args,
        device=device,
        logger=test_logger
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_samples = sum(eval_results["class_total"].values())

    test_logger(f"📦 Number of test samples: {total_samples}")
    test_logger(f"⏱️ Total test time: {elapsed_time:.2f} seconds")
    test_logger("✅ Test evaluation complete.")

    # Combine results
    eval_results["elapsed_time"] = elapsed_time
    eval_results["total_samples"] = total_samples

    return eval_results

#---------------------------------------------------------------------------------------------------------------------------------
def test_inference_model(test_project: str, checkpoint_name: str,test_loader, test_run_number: int = 1, enable_log: bool = True):
    import os
    import time
    import torch
    from utils import get_config, get_config_as_args, get_device, LogToFile, prepare_dataloader
    from Vit import get_current_saved_model_vit

    # Load config and args
    config_path = os.path.join(test_project, "configs.yaml")
    config = get_config(config_path)
    args = get_config_as_args(config_path)
    args.shuffle_before_split = True  # Ensure shuffling is enabled for test dataset
    project_title = config["project_title"]

    # Set up result path and logger
    result_path = os.path.join(config["result_path"], project_title)
    args.result_path = result_path

    log_path = os.path.join(result_path, "Testing")
    os.makedirs(log_path, exist_ok=True)
    log_filename = os.path.join(log_path, f"test_{project_title}_run{test_run_number}.txt")
    test_logger = LogToFile(filename=log_filename, enable=enable_log)

    if enable_log:
        test_logger.clear_log_file()
        test_logger(f"🧪 Inference test evaluation started for project: {project_title} | Test Run: {test_run_number}")
    else:
        test_logger = print
        

    
    # Load the model
    checkpoint_path = os.path.join(result_path, checkpoint_name)
    model = get_current_saved_model_vit(args, decoder=True, checkpoints_path=checkpoint_path)
    test_logger(f"📦 Model loaded from {checkpoint_path}")

    # Move model to device and evaluate
    device = get_device()
    model.to(device)
    model.eval()

    args.print_acc_per_class = False
    start_time = time.time()

    results = evaluate_model(
    model=model,
    dataloader=test_loader,
    criterion=get_criterion(args.loss_function),
    args=args,
    device=device,
    logger=test_logger
)

    elapsed_time = time.time() - start_time
    total_samples = sum(results["class_total"].values())

    test_logger(f"📦 Number of test samples: {total_samples}")
    test_logger(f"⏱️ Total test time: {elapsed_time:.2f} seconds")
    test_logger("✅ Test evaluation complete.")

    # Add timing info to results
    results["elapsed_time"] = elapsed_time
    results["total_samples"] = total_samples

    return results

