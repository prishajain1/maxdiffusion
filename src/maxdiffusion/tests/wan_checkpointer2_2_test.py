"""
 Copyright 2025 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import unittest
from unittest.mock import patch, MagicMock, ANY

from maxdiffusion.checkpointing.wan_checkpointer2_2 import WanCheckpointer, WAN_CHECKPOINT


class WanCheckpointerTest(unittest.TestCase):

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.WanPipeline")
  def test_load_from_diffusers(self, mock_wan_pipeline, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = None
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_pretrained.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    mock_wan_pipeline.from_pretrained.assert_called_once_with(self.config)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertIsNone(step)

  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.epath", autospec=True)
  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.WanPipeline")
  def test_load_checkpoint_no_optimizer(self, mock_wan_pipeline, mock_create_manager, mock_epath): # CHANGED: Added mock_epath
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    metadata_mock.wan_config = {}

    del metadata_mock.optimizer_state
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.get.return_value = None

    mock_manager.restore.return_value = restored_mock
    mock_create_manager.return_value = mock_manager
    mock_epath.Path.return_value = self.config.checkpoint_dir # CHANGED: Mock epath.Path

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(directory=self.config.checkpoint_dir, step=1, args=ANY)
    mock_wan_pipeline.from_checkpoint.assert_called_with(self.config, mock_manager.restore.return_value)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.epath", autospec=True)
  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer2_2.WanPipeline")
  def test_load_checkpoint_with_optimizer(self, mock_wan_pipeline, mock_create_manager, mock_epath): # CHANGED: Added mock_epath
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    metadata_mock.wan_config = {}
    metadata_mock.optimizer_state = {} 
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    optimizer_state_dict = {"learning_rate": 0.001}
    restored_mock.get.side_effect = lambda key, default=None: optimizer_state_dict if key == "optimizer_state" else MagicMock()

    mock_manager.restore.return_value = restored_mock
    mock_create_manager.return_value = mock_manager
    mock_epath.Path.return_value = self.config.checkpoint_dir # CHANGED: Mock epath.Path

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(directory=self.config.checkpoint_dir, step=1, args=ANY)
    mock_wan_pipeline.from_checkpoint.assert_called_with(self.config, mock_manager.restore.return_value)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)
    self.assertEqual(step, 1)


if __name__ == "__main__":
  unittest.main()
