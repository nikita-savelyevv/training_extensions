"""
Time monitor callback module.
"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

# pylint: disable=too-many-instance-attributes,too-many-arguments

import logging
import math
import time
from typing import List, Optional

from ote_sdk.entities.train_parameters import UpdateProgressCallback
from ote_sdk.usecases.reporting.callback import Callback

logger = logging.getLogger(__name__)


class TimeMonitorCallback(Callback):
    """
    A callback to monitor the progress of training.

    :param num_epoch: Amount of epochs
    :param num_train_steps: amount of training steps per epoch
    :param num_val_steps: amount of validation steps per epoch
    :param num_test_steps: amount of testing steps
    :param epoch_history: Amount of previous epochs to calculate average epoch time over
    :param step_history: Amount of previous steps to calculate average steps time over
    """

    def __init__(
        self,
        num_epoch,
        num_train_steps,
        num_val_steps,
        num_test_steps,
        epoch_history: int = 5,
        step_history: int = 50,
        update_progress_callback: Optional[UpdateProgressCallback] = None,
    ):

        self.total_epochs = num_epoch
        self.train_steps = num_train_steps
        self.val_steps = num_val_steps
        self.test_steps = num_test_steps
        self.steps_per_epoch = self.train_steps + self.val_steps
        self.total_steps = math.ceil(
            self.steps_per_epoch * self.total_epochs + num_test_steps
        )
        self.current_step = 0
        self.current_epoch = 0

        # Step time calculation
        self.start_step_time = time.time()
        self.past_step_duration: List[float] = []
        self.average_step = 0
        self.step_history = step_history

        # epoch time calculation
        self.start_epoch_time = time.time()
        self.past_epoch_duration: List[float] = []
        self.average_epoch = 0
        self.epoch_history = epoch_history

        # whether model is training flag
        self.is_training = False

        self.update_progress_callback = update_progress_callback

    def on_train_batch_begin(self, batch, logs=None):
        self.current_step += 1
        self.start_step_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.__calculate_average_step()

    def is_stalling(self) -> bool:
        """
        returns True if the current step has taken more than 30 seconds and
        at least 20x more than the average step duration
        """
        factor = 20
        min_abs_threshold = 30  # seconds
        if self.is_training and self.current_step > 2:
            step_duration = time.time() - self.start_step_time
            if (
                step_duration > min_abs_threshold
                and step_duration > factor * self.average_step
            ):
                logger.error(
                    f"Step {self.current_step} has taken {step_duration}s which is "
                    f">{min_abs_threshold}s and  {factor} times "
                    f"more than the expected {self.average_step}s"
                )
                return True
        return False

    def __calculate_average_step(self):
        self.past_step_duration.append(time.time() - self.start_step_time)
        if len(self.past_step_duration) > self.step_history:
            self.past_step_duration.remove(self.past_step_duration[0])
        self.average_step = sum(self.past_step_duration) / len(self.past_step_duration)

    def on_test_batch_begin(self, batch, logs):
        self.current_step += 1
        self.start_step_time = time.time()

    def on_test_batch_end(self, batch, logs):
        self.__calculate_average_step()

    def on_train_begin(self, logs=None):
        self.is_training = True

    def on_train_end(self, logs=None):
        # To handle cases where early stopping stops the task the progress will still be accurate
        self.current_step = self.total_steps - self.test_steps
        self.current_epoch = self.total_epochs
        self.is_training = False

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.start_epoch_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        self._calculate_average_epoch()
        self.update_progress_callback(self.get_progress())

    def _calculate_average_epoch(self):
        if len(self.past_epoch_duration) > self.epoch_history:
            del self.past_epoch_duration[0]
        self.average_epoch = sum(self.past_epoch_duration) / len(
            self.past_epoch_duration
        )

    def get_progress(self):
        """
        Returns current progress as a percentage.
        """
        return (self.current_step / self.total_steps) * 100
