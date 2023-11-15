import os
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf


def load_file_data(file_path: Path) -> np.ndarray:
    data = None
    try:
        data = np.genfromtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"Error loading the data file: {e}")
        print(f"Ensure that {file_path.name} is in data directory")
    return data


class PathsManager:
    def __init__(self):
        try:
            self.config = OmegaConf.load(Path(__file__).parent / "config.yaml")
        except Exception as e:
            print(f"Error loading the configuration file: {e}")
            print("Please create a 'config.yaml' file with the required configuration.")

        self._root_path = Path(__file__).parent.parent

    @staticmethod
    def _ensure_directory(directory_path: Path) -> Path:
        if not directory_path.exists():
            os.makedirs(directory_path)
        return directory_path

    def data_dir(self) -> Path:
        path = self._root_path / self.config.data_dir
        return self._ensure_directory(path)

    def reports_dir(self) -> Path:
        path = self._root_path / self.config.reports_dir
        return self._ensure_directory(path)
