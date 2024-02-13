import engineer

from engineer.schedulers.cosine import CosineAnnealingLR
from torch_geometric.loader import DataLoader

def main(config):
    dataset_config = config["dataset"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)

    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()
    model_config = config["model"]
    model = engineer.load_module(model_config.pop("module"))(**model_config)

    model = model.cuda()
    optimizer_config = config["optimizer"]
    optimizer = engineer.load_module(optimizer_config.pop("module"))(
        model.parameters(), **optimizer_config
    )

    # scheduler_config = config['scheduler']
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode=scheduler_config["mode"], 
    #     factor=scheduler_config["factor"], 
    #     patience=scheduler_config["patience"])

    steps = config["trainer"]["max_steps"]
    scheduler = CosineAnnealingLR(
        optimizer,
        steps,
        warmup_steps=int(1 / 64 * steps),
        decay_steps=int(1 / 4 * steps),
    )

    trainer_module = engineer.load_module(config["trainer"].pop("module"))

    trainer_config = config["trainer"]
    trainer_config["scheduler"] = scheduler
    # trainer_config["wandb"] = config["wandb"]
    trainer = trainer_module(
        **trainer_config,
    )
    trainer.fit(model, optimizer, train_loader, val_loader=val_loader, test_loader=test_loader)


if __name__ == "__main__":
    engineer.fire(main)
