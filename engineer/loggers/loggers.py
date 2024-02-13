import os

import torch


class WANDBLogger:
    def __init__(self):
        if torch.distributed.is_initialized():
            assert (
                torch.distributed.get_rank() == 0
            ), "WANDBLogger should only be initialized on rank 0."

        self.metrics = set()
        self.dir = wandb.run.dir

    @property
    def initialized(self):
        return wandb.run is not None

    def _log(self, dict, step):
        if not self.initialized:
            return
        wandb.log(dict, step=step)

    def log_metrics(self, metrics, step):
        if not self.initialized:
            return

        for m in metrics:
            if m not in self.metrics:
                wandb.define_metric(m, summary="max,min,last")
                print(f"Defined metric {m}.")
                self.metrics.add(m)

        return self._log(metrics, step)

    def log_image(self, image_dict, step):
        image_dict = {k: wandb.Image(v) for k, v in image_dict.items()}
        return self._log(image_dict, step)

    def save(self, file):
        if not self.initialized:
            print("Not saving because WANDB is not initialized.")
            return
        wandb.save(file, base_path=os.path.dirname(file))

    def save_model(self, file, alias):
        name = str(wandb.run.id) + "-" + "model"
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(file)
        wandb.log_artifact(artifact, aliases=[alias])

        project = wandb.run.project
        entity = wandb.run.entity
        id = wandb.run.id
        run = wandb.Api().run(f"{entity}/{project}/{id}")

        for v in run.logged_artifacts():
            if len(v.aliases) == 0:
                v.delete()


def _pp(d, indent=0):
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            _pp(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


class ConsoleLogger:
    def __init__(self) -> None:
        self.metrics = []
        self.dir = None

    def _log(self, dict, step):
        # Print metrics
        print()
        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            print(f"{k}: {v:.4f}")
        print()

    def log_metrics(self, metrics, step):
        for m in metrics:
            if m not in self.metrics:
                print(f"Defined metric {m}.")
                self.metrics.append(m)

        return self._log(metrics, step)
