from utils import *
from dataset  import *
from dataset import *
from dataset_finger import *
from augementation_iq import *
from Vit import *
from trainer import *
import itertools
import torch
from plotting_configs import *
from plotting_factory import *
from evaluation import *
import itertools
import os
import pickle
import time
#-------------------------------------------------

config = get_config()
device = get_device()
args = get_config_as_args()
project_title=config["project_title"]
result_path=os.path.join(config["result_path"], project_title)


args.result_path = result_path
logger = LogToFile(filename=f"{result_path}/outputs_{project_title}.txt", enable=True)
logger.clear_log_file()

main_dataset = RML2018_Dataset( args=args,logger=logger)
main_dataset.summarize_dataset()
logger(f' {project_title} Training started ---------------------------------------------------------------------\n')  

for key, value in config.items():
    logger(f"{key}: {value}")
logger(f"-------------------------------------------------------------------------------\n")  
file_list = config.get("files_to_copy", [])
copy_files_to_path(file_list=file_list, destination_path=result_path) 
# Initialize model and trainer -------------------------------------------------------------


def train_VitModel_with_grid_search():
    # Define a grid of hyperparameters
    hyperparameter_grid = args.grid_search_vit

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for idx, config_model in enumerate(hyperparameter_combinations, start=1):
        logger(f"\n🧪 Training {idx}: with hyperparameters: {config_model} **********")
        if args.shuffle_before_split :
           main_dataset.shuffle_and_restart()
           main_dataset.summarize_dataset()
        # Prepare dataloaders
        val_loader = prepare_dataloader(main_dataset.val_set, config)
        train_loader = prepare_dataloader(main_dataset.train_set, config)
        test_loader = prepare_dataloader(main_dataset.test_set, config)

        # Load checkpoint if provided
        checkpoints_path = config["pretrained_path"]
        model = get_current_saved_model_vit(args, decoder=True, checkpoints_path=checkpoints_path,config_model=config_model)
        
        quick_plot_dir = os.path.join(result_path, f"Quick_plot_{idx}")
        os.makedirs(quick_plot_dir, exist_ok=True)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            config_model=config_model,
            args=args,
            logger_main=logger,
            val_loader=val_loader,
            train_loader=train_loader,
            result_path=quick_plot_dir,
            run_number=idx
        )

        logger("⏱️ Starting training...")

        # ⏳ Time the training + validation
        start_time = time.time()
        result = trainer.train()
        metrics = result["metrics"]
        model = result["model"]
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60.0
        logger(f"✅ Training {idx} completed in {elapsed_minutes:.2f} minutes.")

        # Save model
        model_name = f"model_{project_title.replace(' ', '_')}_run{idx}.pth"
        save_path = os.path.join(result_path, model_name)
        save_model(model, save_path)
        logger(f"📦 Model saved to {save_path}")

        # Create unique Quick_plot folder and save metrics
        
        pkl_name = f"metrics_{project_title.replace(' ', '_')}_run{idx}.pkl"
        pkl_path = os.path.join(quick_plot_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(metrics, f)
        logger(f"📊 Metrics saved to {pkl_path}")

        # Plot metrics
        key_groups = [
            ["train_loss","val_loss"],
            ["train_loss_constrative", "train_loss_reconstruction", "train_loss_classification"],
            ["train_accuracy", "val_accuracy"],
        ]
        plot_config__paper_metrice["save_path"] = quick_plot_dir
        plot_metrics_to_pdf(pkl_path, f"{pkl_path}.pdf", key_groups, plot_config__paper_metrice)
        logger(f"📈 Metrics plot saved to {pkl_path}.pdf")

        # Run test evaluation
        logger(f"🔍 Test model {idx} on test data loader set...")
        test_model(model, idx, test_loader, args,config_model)

#------------------------------------

#------------------------------------------

train_VitModel_with_grid_search()
