import numpy as np
from torch.utils.data import Sampler


class _BalancedDownsampler(Sampler[int]):
    """
    Downsamples the majority class to match the minority class count.
    A different majority-class subset is drawn on every epoch, so across
    training the model eventually sees all of the majority data.

    Assumes binary labels.
    """

    def __init__(self, labels, seed=None):
        labels = np.asarray(labels)

        classes, counts = np.unique(labels, return_counts=True)
        minority_class = classes[np.argmin(counts)]

        self.min_idx = np.arange(len(labels))[labels == minority_class]
        self.maj_idx = np.arange(len(labels))[labels != minority_class]
        self.gen = np.random.default_rng(seed)

    def __iter__(self):
        maj = self.gen.permutation(self.maj_idx)[:len(self.min_idx)]
        idx = np.concatenate([maj, self.min_idx])
        self.gen.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return 2 * len(self.min_idx)


def build_downsampler(labels, seed=None):
    """Build a sampler that rebalances the classes, resampling each epoch."""
    return _BalancedDownsampler(labels, seed=seed)


if __name__ == '__main__':
    # Example usage
    records = [
        {'id': 0, 'label': 0},
        {'id': 1, 'label': 1},
        {'id': 2, 'label': 0},
        {'id': 3, 'label': 1},
        {'id': 4, 'label': 1},
    ]
    labels = np.array([r['label'] for r in records])
    sampler = build_downsampler(labels)

    for epoch in range(3):
        print(f"epoch {epoch}: {[records[i]['id'] for i in sampler]}")