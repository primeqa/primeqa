import time
import numpy as np
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def time_str(seconds: float) -> str:
    if seconds > 60 * 60:
        return f'{seconds/(60.0*60.0):.1f} hours'
    if seconds > 60:
        return f'{seconds/60.0:.1f} minutes'
    return f'{int(seconds)} seconds'


class Reporting:
    def __init__(self, *, recency_weight=0.001, report_interval_secs=300, check_every=1,
                 gather_samples:Iterable=(), num_samples=10000):
        """

        :param recency_weight: when computing the moving average, how much weight to give to the current sample
        :param report_interval_secs: how many seconds between returning true for is_time
        :param check_every: how often to check the time, when calling is_time
        :param gather_samples: keep the last num_samples of the listed names (gathered from moving_averages)
        :param num_samples: how many samples to keep
        """
        self.check_count = 0
        self.check_every = check_every
        self.start_time = time.time()
        self.last_time = self.start_time
        self.report_interval_secs = report_interval_secs
        # For tracking moving averages of various values
        self.names = None
        self.averages = None
        self.counts = None
        self.recency_weight = recency_weight
        self.per_value_recency_weight = dict()
        self.report_count = 0
        self._prev_check_count = 0
        self.sample_names = list(gather_samples)
        if len(self.sample_names) > 0:
            self.sample_values = np.zeros((len(self.sample_names), num_samples), dtype=np.float32)
            self.sample_ndxs = np.zeros(len(self.sample_names), dtype=np.int32)
        else:
            self.sample_values = None
            self.sample_ndxs = None

    def reset(self):
        self.check_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.report_count = 0
        self._prev_check_count = 0
        if len(self.sample_names) > 0:
            self.sample_values[:, :] = 0
            self.sample_ndxs[:] = 0
        if self.counts is not None:
            self.counts[:] = 0
            self.averages[:] = 0

    def is_time(self):
        self.check_count += 1
        if self.check_count % self.check_every == 0:
            elapsed = time.time() - self.last_time
            if elapsed >= self.report_interval_secs:
                # check the time more or less often
                if self.check_every > 1 and self.check_count - self._prev_check_count < 5 * self.check_every:
                    self.check_every //= 2
                elif self.check_count - self._prev_check_count > 50 * self.check_every:
                    self.check_every *= 2
                self.last_time = time.time()
                self.report_count += 1
                self._prev_check_count = self.check_count
                return True
        return False

    def moving_averages(self, **values):
        # create entries in avgs and counts when needed
        # update the avgs and counts
        if self.names is None:
            self.names = list(values.keys())
            self.averages = np.zeros(len(self.names))
            self.counts = np.zeros(len(self.names), dtype=np.int32)
        for name in values.keys():
            if name not in self.names:
                self.names.append(name)
        if self.averages.shape[0] < len(self.names):
            old_len = self.averages.shape[0]
            self.averages = np.resize(self.averages, len(self.names))
            self.averages[old_len:] = 0
            self.counts = np.resize(self.counts, len(self.names))
            self.counts[old_len:] = 0
        for ndx, name in enumerate(self.names):
            if name in values:
                self.counts[ndx] += 1
                # support per-name recency_weight
                if name in self.per_value_recency_weight:
                    rweight = max(self.per_value_recency_weight[name], 1.0 / self.counts[ndx])
                else:
                    rweight = max(self.recency_weight, 1.0 / self.counts[ndx])
                self.averages[ndx] = rweight * values[name] + (1.0 - rweight) * self.averages[ndx]
        for ndx, name in enumerate(self.sample_names):
            if name in values:
                self.sample_values[ndx, self.sample_ndxs[ndx]] = values[name]
                self.sample_ndxs[ndx] = (self.sample_ndxs[ndx] + 1) % self.sample_values.shape[1]

    def get_samples(self, name: str):
        for ndx, n in enumerate(self.sample_names):
            if n == name:
                count = self.get_count(name)
                if count is None:
                    count = 0
                return self.sample_values[ndx, 0:count]  # NOTE: not in order
        return None

    def get_moving_average(self, name):
        if self.names is None:
            return None
        for ndx, n in enumerate(self.names):
            if n == name:
                return self.averages[ndx]
        return None

    def get_count(self, name):
        if self.names is None:
            return None
        for ndx, n in enumerate(self.names):
            if n == name:
                return self.counts[ndx]
        return None

    def elapsed_seconds(self) -> float:
        return time.time()-self.start_time

    def elapsed_time_str(self) -> str:
        return time_str(self.elapsed_seconds())

    def progress_str(self, instance_name='instance'):
        return f'On {instance_name} {self.check_count}, ' \
               f'{self.check_count/self.elapsed_seconds()} {instance_name}s per second.'

    def display(self, *, prefix=''):
        # display the moving averages
        logger.info('==========================================')
        if self.names is not None:
            for n, v in zip(self.names, self.averages):
                logger.info(f'{prefix}{n} = {v}')

    def display_warn(self, *, prefix=''):
        # display the moving averages
        logger.info('==========================================')
        if self.names is not None:
            for n, v in zip(self.names, self.averages):
                logger.warning(f'{prefix}{n} = {v}')
