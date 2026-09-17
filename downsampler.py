import numpy as np
from torch.utils.data import Sampler


class _BalancedDownsampler(Sampler[int]):
    """
    Draws an equal number of indices from every class, redrawn on every epoch,
    so across training the model eventually sees all of the data.

    per_class caps that draw. Left at None it is the minority-class count, which
    is the largest balanced draw the labels support — that is the single-site
    behaviour. Passing a smaller value (the federated cap, so every site trains
    on the same number of images per epoch) shrinks both classes to match.
    """

    def __init__(self, labels, per_class=None, seed=None):
        labels = np.asarray(labels)

        classes = np.unique(labels)
        self.class_idx = [np.flatnonzero(labels == c) for c in classes]

        largest_balanced = min(len(ix) for ix in self.class_idx)
        self.per_class = (
            largest_balanced if per_class is None
            else min(int(per_class), largest_balanced)
        )
        self.gen = np.random.default_rng(seed)

    def __iter__(self):
        idx = np.concatenate([
            self.gen.permutation(ix)[:self.per_class] for ix in self.class_idx
        ])
        self.gen.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return self.per_class * len(self.class_idx)


def build_downsampler(labels, per_class=None, seed=None):
    """Build a sampler that rebalances the classes, resampling each epoch."""
    return _BalancedDownsampler(labels, per_class=per_class, seed=seed)


def min_class_count(labels):
    """Largest per-class draw these labels support, i.e. the minority count."""
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    return int(counts.min())


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
    print(f"uncapped, per_class={sampler.per_class}, epoch size {len(sampler)}")
    for epoch in range(3):
        print(f"  epoch {epoch}: {[records[i]['id'] for i in sampler]}")

    capped = build_downsampler(labels, per_class=1)
    print(f"capped, per_class={capped.per_class}, epoch size {len(capped)}")
    for epoch in range(3):
        print(f"  epoch {epoch}: {[records[i]['id'] for i in capped]}")
