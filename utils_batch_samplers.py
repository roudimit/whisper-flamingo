# copied from ESPnet
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, Tuple

class SortedBatchSampler(Sampler):
    """BatchSampler with sorted samples by length.

    Args:
        batch_size:
        shape_file:
        sort_in_batch: 'descending', 'ascending' or None.
        sort_batch:
    """

    def __init__(
        self,
        batch_size: int,
        shapes: list,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
    ):
        assert batch_size > 0
        self.batch_size = batch_size
        # self.shape_file = shape_file
        self.shapes = shapes
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        # utt2shape = load_num_sequence_text(shape_file, loader_type="csv_int")
        utt2shape = {i: shapes[i] for i in range(len(shapes))}
        if sort_in_batch == "descending":
            # Sort samples in descending order (required by RNN)
            keys = sorted(utt2shape, key=lambda k: -utt2shape[k])
        elif sort_in_batch == "ascending":
            # Sort samples in ascending order
            keys = sorted(utt2shape, key=lambda k: utt2shape[k])
        else:
            raise ValueError(
                f"sort_in_batch must be either one of "
                f"ascending, descending, or None: {sort_in_batch}"
            )
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shapes}")

        # Apply max(, 1) to avoid 0-batches
        N = max(len(keys) // batch_size, 1)
        if not self.drop_last:
            # Split keys evenly as possible as. Note that If N != 1,
            # the these batches always have size of batch_size at minimum.
            self.batch_list = [
                keys[i * len(keys) // N : (i + 1) * len(keys) // N] for i in range(N)
            ]
        else:
            self.batch_list = [
                tuple(keys[i * batch_size : (i + 1) * batch_size]) for i in range(N)
            ]

        if len(self.batch_list) == 0:
            print(f"{shapes} is empty")

        if sort_in_batch != sort_batch:
            if sort_batch not in ("ascending", "descending"):
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.reverse()

        if len(self.batch_list) == 0:
            raise RuntimeError("0 batches")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)


class LengthBatchSampler(Sampler):
    def __init__(
        self,
        batch_bins: int,
        # shape_files: Union[Tuple[str, ...], List[str]],
        shapes: list,
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert batch_bins > 0
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        # self.shape_files = shape_files
        self.shapes = shapes
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        utt2shapes = [{i: shapes[i] for i in range(len(shapes))}]
        first_utt2shape = utt2shapes[0]

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k])

        # Decide batch-sizes
        batch_sizes = []
        current_batch_keys = []
        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if padding:
                # bins = bs x max_length
                bins = sum(len(current_batch_keys) * sh[key] for sh in utt2shapes)
            else:
                # bins = sum of lengths
                bins = sum(d[k] for k in current_batch_keys for d in utt2shapes)

            if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        # Set mini-batch
        self.batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {sort_in_batch}"
                    )
                self.batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        elif sort_batch == "shuffle":
            np.random.shuffle(self.batch_list)
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)