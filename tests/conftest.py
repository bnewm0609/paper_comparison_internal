import pytest

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

# @pytest.fixture(scope="session")
# def template_data_test_config():
#     """Returns the testing config"""
#     with initialize(version_base=None, config_path="fixtures/configs"):
#         cfg = compose(
#             config_name="config", overrides=["data=test_template", "model=test_bart", "wandb=False"]
#         )
#         return cfg
