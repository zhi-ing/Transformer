import os
import tempfile
import unittest

import torch
import torch.nn as nn

from utils.checkpoint import load_checkpoint, save_checkpoint


class DummyScheduler:
    def __init__(self):
        self.step_count = 0

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]

    def step(self):
        self.step_count += 1


class CheckpointResumeTest(unittest.TestCase):
    def test_save_and_load_checkpoint_restores_epoch_and_best_loss(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = DummyScheduler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "resume.pth")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=5,
                path=path,
                best_val_loss=0.25,
            )

            reloaded_model = nn.Linear(2, 1)
            reloaded_optimizer = torch.optim.SGD(reloaded_model.parameters(), lr=0.0)
            reloaded_scheduler = DummyScheduler()

            epoch = load_checkpoint(
                path,
                reloaded_model,
                reloaded_optimizer,
                reloaded_scheduler,
                device="cpu",
            )

            self.assertEqual(epoch, 5)
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["best_val_loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
