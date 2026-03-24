from dagster import ConfigurableResource


class ArtifactStorage(ConfigurableResource):
    base_path: str = "model/checkpoints"

    def checkpoint_path(self, name: str = "best_model.pt") -> str:
        import os
        os.makedirs(self.base_path, exist_ok=True)
        return os.path.join(self.base_path, name)
