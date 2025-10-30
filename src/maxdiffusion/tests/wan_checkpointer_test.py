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
from unittest.mock import patch, MagicMock

from maxdiffusion.checkpointing.wan_checkpointer import (
    WanCheckpointer2_1, 
    WanCheckpointer2_2, 
    WAN_CHECKPOINT
)


class WanCheckpointer2_1Test(unittest.TestCase):
  """Tests for WAN 2.1 checkpointer."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_1")
  def test_load_from_diffusers(self, mock_wan_pipeline, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = None
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_pretrained.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    mock_wan_pipeline.from_pretrained.assert_called_once_with(self.config)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertIsNone(step)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_1")
  def test_load_checkpoint_no_optimizer(self, mock_wan_pipeline, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["wan_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(directory=unittest.mock.ANY, step=1, args=unittest.mock.ANY)
    mock_wan_pipeline.from_checkpoint.assert_called_with(self.config, mock_manager.restore.return_value)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_1")
  def test_load_checkpoint_with_optimizer(self, mock_wan_pipeline, mock_create_manager):
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["wan_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(directory=unittest.mock.ANY, step=1, args=unittest.mock.ANY)
    mock_wan_pipeline.from_checkpoint.assert_called_with(self.config, mock_manager.restore.return_value)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  def test_save_checkpoint(self, mock_create_manager):
    """Test saving checkpoint for WAN 2.1."""
    mock_manager = MagicMock()
    mock_create_manager.return_value = mock_manager

    mock_pipeline = MagicMock()
    mock_pipeline.transformer.to_json_string.return_value = '{"config": "test"}'

    train_states = {"params": {}, "opt_state": {}}

    checkpointer = WanCheckpointer2_1(self.config, WAN_CHECKPOINT)
    checkpointer.save_checkpoint(train_step=100, pipeline=mock_pipeline, train_states=train_states)

    mock_manager.save.assert_called_once()
    call_args = mock_manager.save.call_args
    self.assertEqual(call_args[0][0], 100)  # train_step


class WanCheckpointer2_2Test(unittest.TestCase):
  """Tests for WAN 2.2 checkpointer."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_2_2_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_2")
  def test_load_from_diffusers(self, mock_wan_pipeline, mock_create_manager):
    """Test loading from pretrained when no checkpoint exists."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = None
    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_pretrained.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    mock_wan_pipeline.from_pretrained.assert_called_once_with(self.config)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertIsNone(step)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_2")
  def test_load_checkpoint_no_optimizer(self, mock_wan_pipeline, mock_create_manager):
    """Test loading checkpoint without optimizer state."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}}
    restored_mock.high_noise_transformer_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(directory=unittest.mock.ANY, step=1, args=unittest.mock.ANY)
    mock_wan_pipeline.from_checkpoint.assert_called_with(self.config, mock_manager.restore.return_value)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNone(opt_state)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_2")
  def test_load_checkpoint_with_optimizer_in_low_noise(self, mock_wan_pipeline, mock_create_manager):
    """Test loading checkpoint with optimizer state in low_noise_transformer."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.high_noise_transformer_state = {"params": {}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(directory=unittest.mock.ANY, step=1, args=unittest.mock.ANY)
    mock_wan_pipeline.from_checkpoint.assert_called_with(self.config, mock_manager.restore.return_value)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_2")
  def test_load_checkpoint_with_optimizer_in_high_noise(self, mock_wan_pipeline, mock_create_manager):
    """Test loading checkpoint with optimizer state in high_noise_transformer."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}}
    restored_mock.high_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.002}}
    restored_mock.wan_config = {}
    restored_mock.keys.return_value = ["low_noise_transformer_state", "high_noise_transformer_state", "wan_config"]

    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    mock_manager.restore.assert_called_once_with(directory=unittest.mock.ANY, step=1, args=unittest.mock.ANY)
    mock_wan_pipeline.from_checkpoint.assert_called_with(self.config, mock_manager.restore.return_value)
    self.assertEqual(pipeline, mock_pipeline_instance)
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.002)
    self.assertEqual(step, 1)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  def test_save_checkpoint(self, mock_create_manager):
    """Test saving checkpoint for WAN 2.2."""
    mock_manager = MagicMock()
    mock_create_manager.return_value = mock_manager

    mock_pipeline = MagicMock()
    mock_pipeline.low_noise_transformer.to_json_string.return_value = '{"config": "test"}'

    train_states = {
        "low_noise_transformer": {"params": {}, "opt_state": {}},
        "high_noise_transformer": {"params": {}}
    }

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    checkpointer.save_checkpoint(train_step=100, pipeline=mock_pipeline, train_states=train_states)

    mock_manager.save.assert_called_once()
    call_args = mock_manager.save.call_args
    self.assertEqual(call_args[0][0], 100)  # train_step

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  def test_save_checkpoint_validates_train_states(self, mock_create_manager):
    """Test that save_checkpoint expects both transformer states."""
    mock_manager = MagicMock()
    mock_create_manager.return_value = mock_manager

    mock_pipeline = MagicMock()
    mock_pipeline.low_noise_transformer.to_json_string.return_value = '{"config": "test"}'

    train_states = {
        "low_noise_transformer": {"params": {}},
        "high_noise_transformer": {"params": {}}
    }

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    
    # This should not raise an error
    checkpointer.save_checkpoint(train_step=100, pipeline=mock_pipeline, train_states=train_states)
    mock_manager.save.assert_called_once()


class WanCheckpointerFactoryTest(unittest.TestCase):
  """Tests for checkpointer factory/selection logic."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_factory_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  def test_create_correct_checkpointer_2_1(self, mock_create_manager):
    """Test creating WAN 2.1 checkpointer."""
    mock_manager = MagicMock()
    mock_create_manager.return_value = mock_manager

    checkpointer = WanCheckpointer2_1(self.config, WAN_CHECKPOINT)
    
    self.assertIsInstance(checkpointer, WanCheckpointer2_1)
    self.assertEqual(checkpointer.config, self.config)
    self.assertEqual(checkpointer.checkpoint_type, WAN_CHECKPOINT)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  def test_create_correct_checkpointer_2_2(self, mock_create_manager):
    """Test creating WAN 2.2 checkpointer."""
    mock_manager = MagicMock()
    mock_create_manager.return_value = mock_manager

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    
    self.assertIsInstance(checkpointer, WanCheckpointer2_2)
    self.assertEqual(checkpointer.config, self.config)
    self.assertEqual(checkpointer.checkpoint_type, WAN_CHECKPOINT)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  def test_checkpointers_have_different_implementations(self, mock_create_manager):
    """Test that 2.1 and 2.2 checkpointers are different classes."""
    mock_manager = MagicMock()
    mock_create_manager.return_value = mock_manager

    checkpointer_2_1 = WanCheckpointer2_1(self.config, WAN_CHECKPOINT)
    checkpointer_2_2 = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    
    self.assertNotEqual(type(checkpointer_2_1), type(checkpointer_2_2))
    self.assertTrue(hasattr(checkpointer_2_1, 'load_checkpoint'))
    self.assertTrue(hasattr(checkpointer_2_2, 'load_checkpoint'))
    self.assertTrue(hasattr(checkpointer_2_1, 'save_checkpoint'))
    self.assertTrue(hasattr(checkpointer_2_2, 'save_checkpoint'))


class WanCheckpointerEdgeCasesTest(unittest.TestCase):
  """Tests for edge cases and error handling."""

  def setUp(self):
    self.config = MagicMock()
    self.config.checkpoint_dir = "/tmp/wan_checkpoint_edge_test"
    self.config.dataset_type = "test_dataset"

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_1")
  def test_load_checkpoint_with_explicit_none_step(self, mock_wan_pipeline, mock_create_manager):
    """Test loading checkpoint with explicit None step falls back to latest."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 5
    metadata_mock = MagicMock()
    metadata_mock.wan_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.wan_state = {"params": {}}
    restored_mock.wan_config = {}
    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_1(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=None)

    mock_manager.latest_step.assert_called_once()
    self.assertEqual(step, 5)

  @patch("maxdiffusion.checkpointing.wan_checkpointer.create_orbax_checkpoint_manager")
  @patch("maxdiffusion.checkpointing.wan_checkpointer.WanPipeline2_2")
  def test_load_checkpoint_both_optimizers_present(self, mock_wan_pipeline, mock_create_manager):
    """Test loading checkpoint when both transformers have optimizer state (prioritize low_noise)."""
    mock_manager = MagicMock()
    mock_manager.latest_step.return_value = 1
    metadata_mock = MagicMock()
    metadata_mock.low_noise_transformer_state = {}
    metadata_mock.high_noise_transformer_state = {}
    mock_manager.item_metadata.return_value = metadata_mock

    restored_mock = MagicMock()
    restored_mock.low_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.001}}
    restored_mock.high_noise_transformer_state = {"params": {}, "opt_state": {"learning_rate": 0.002}}
    restored_mock.wan_config = {}
    mock_manager.restore.return_value = restored_mock

    mock_create_manager.return_value = mock_manager

    mock_pipeline_instance = MagicMock()
    mock_wan_pipeline.from_checkpoint.return_value = mock_pipeline_instance

    checkpointer = WanCheckpointer2_2(self.config, WAN_CHECKPOINT)
    pipeline, opt_state, step = checkpointer.load_checkpoint(step=1)

    # Should prioritize low_noise_transformer's optimizer state
    self.assertIsNotNone(opt_state)
    self.assertEqual(opt_state["learning_rate"], 0.001)


if __name__ == "__main__":
  unittest.main(verbosity=2)
