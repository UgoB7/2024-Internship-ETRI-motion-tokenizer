# Hackathon template
A convenient starting template for most deep learning projects. Built with <b>PyTorch Lightning</b> and <b>Weights&Biases</b>.
<br>


## Setup
Read [SETUP.md](SETUP.md)
<br><br>


## Project structure
The directory structure of new project looks like this: 
```
├── project
│   ├── data                            <- Data from third party sources
│   │
│   ├── logs                    <- Logs generated by Weights&Biases and PyTorch Lightning
│   │
│   ├── notebooks               <- Jupyter notebooks
│   │
│   ├── utils                   <- Different utilities
│   │   ├── callbacks.py                <- Useful training callbacks
│   │   ├── execute_sweep.py            <- Special file for executing Weights&Biases sweeps
│   │   ├── init_utils.py               <- Useful initializers
│   │   └── predict_example.py          <- Example of inference with trained model 
│   │
│   ├── data_modules            <- All your data modules should be located here
│   │   ├── example_datamodule          <- Each datamodule should be located in separate folder!
│   │   │   ├── datamodule.py                   <- Contains 'DataModule' class
│   │   │   ├── datasets.py                     <- Contains pytorch 'Dataset' classes
│   │   │   └── transforms.py                   <- Contains data transformations
│   │   ├── ...
│   │   └── ...
│   │
│   ├── models                  <- All your models should be located here
│   │   ├── example_model               <- Each model should be located in separate folder!
│   │   │   ├── lightning_module.py             <- Contains 'LitModel' class with train/val/test step methods
│   │   │   └── models.py                       <- Model architectures used by lightning_module.py
│   │   ├── ...
│   │   └── ...
│   │
│   ├── project_config.yaml     <- Project configuration
│   ├── run_configs.yaml        <- Configurations of different runs/experiments
│   └── train.py                <- Train model with chosen run configuration
│
├── .gitignore
├── LICENSE
├── README.md
├── SETUP.md
└── requirements.txt
```

## Project config parameters ([project_config.yaml](project/project_config.yaml))
```yaml
num_of_gpus: -1

resume_training:
    lightning_ckpt:
        resume_from_ckpt: False
        ckpt_path: "logs/checkpoints/epoch=2.ckpt"
    wandb:
        resume_wandb_run: False
        wandb_run_id: "8uuomodb"

loggers:
    wandb:
        project: "hackathon_template_test"
        team: "kino"
        group: None
        job_type: "train"
        tags: []
        log_model: True
        offline: False

callbacks:
    checkpoint:
        monitor: "val_acc"
        save_top_k: 1
        save_last: True
        mode: "max"
    early_stop:
        monitor: "val_acc"
        patience: 100
        mode: "max"

printing:
    progress_bar_refresh_rate: 5
    weights_summary: "top"  # "full"
    profiler: False
```

## Run config parameters ([run_configs.yaml](project/run_configs.yaml))
You can store many run configurations in this file.<br>
Example run configuration:
```yaml
MNIST_CLASSIFIER_V1:
    trainer:
        min_epochs: 1
        max_epochs: 5
        gradient_clip_val: 0.5
        accumulate_grad_batches: 1
        limit_train_batches: 1.0
    model:
        model_folder: "simple_mnist_classifier"
        lr: 0.001
        weight_decay: 0.000001
        input_size: 784  # img size is 1*28*28
        output_size: 10  # there are 10 digit classes
        lin1_size: 256
        lin2_size: 256
        lin3_size: 128
    dataset:
        datamodule_folder: "mnist_digits_datamodule"
        batch_size: 256
        train_val_split_ratio: 0.9
        num_workers: 4
        pin_memory: True
```
The run configuration that you want to train with needs to be chosen in [train.py](project/train.py):
```python
if __name__ == "__main__":
    RUN_CONFIG_NAME = "MNIST_CLASSIFIER_V1"
    main(run_config_name=RUN_CONFIG_NAME)

```
