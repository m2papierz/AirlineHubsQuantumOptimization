import os

from pathlib import Path
from omegaconf import OmegaConf


class PathsManager:
    def __init__(self):
        try:
            self.config = OmegaConf.load(Path(__file__).parent / "config.yaml")
        except Exception as e:
            print(f"Error loading the configuration file: {e}")
            print("Please create a 'config.yaml' file with the required configuration.")

        self.root_path = Path(__file__).parent.parent

    @staticmethod
    def _ensure_directory(directory_path: Path) -> Path:
        if not directory_path.exists():
            os.makedirs(directory_path)
        return directory_path

    def data_dir(self) -> Path:
        path = self.root_path / self.config.data_dir
        return self._ensure_directory(path)
