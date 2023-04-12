import unittest

import torch
from uvcgan2.torch.queue import Queue, FastQueue

class TestQueues(unittest.TestCase):

    # pylint: disable=protected-access
    def _compare_fast_queue_to_sequence(self, fast_queue, sequence):
        self.assertEqual(sum(len(x) for x in sequence), len(fast_queue))

        if len(sequence) == 0:
            return

        fast_queue_tensor = fast_queue.query()

        if len(fast_queue) != fast_queue._size:
            fast_queue_idx = 0
        else:
            fast_queue_idx = fast_queue._curr_idx

        for item in sequence:
            for item_idx in range(len(item)):
                fast_queue_idx = fast_queue_idx % len(fast_queue)
                fast_item \
                    = fast_queue_tensor[fast_queue_idx:fast_queue_idx+1, ...]

                self.assertTrue(
                    (fast_item == item[item_idx:item_idx+1, ...]).all()
                )

                fast_queue_idx += 1

    def _compare_queue_to_sequence(self, queue, sequence):
        self.assertEqual(len(sequence), len(queue))

        if len(queue) == 0:
            return

        queue_seq = queue.query()

        for (s1, s2) in zip(sequence, queue_seq):
            self.assertTrue((s1 == s2).all())

    def _compare_queues(self, queue, fast_queue):
        self.assertEqual(len(queue), len(fast_queue))

        if len(queue) == 0:
            return

        self._compare_fast_queue_to_sequence(fast_queue, queue.query())

    def test_queue_simple(self):
        sequence = [ torch.randn((1, 2, 3, 4)) for _ in range(10) ]
        queue    = Queue(len(sequence))

        for item in sequence:
            queue.push(item)

        self._compare_queue_to_sequence(queue, sequence)

    def test_fast_queue_simple(self):
        sequence = [ torch.randn((1, 2, 3, 4)) for _ in range(10) ]
        queue    = FastQueue(len(sequence), 'cpu')

        for item in sequence:
            queue.push(item)

        self._compare_fast_queue_to_sequence(queue, sequence)

    def test_queue_wrap(self):
        sequence = [ torch.randn((1, 2, 3, 4)) for _ in range(10) ]
        queue    = Queue(5)

        for item in sequence:
            queue.push(item)

        self._compare_queue_to_sequence(queue, sequence[5:])

    def test_fast_queue_wrap(self):
        sequence = [ torch.randn((1, 2, 3, 4)) for _ in range(10) ]
        queue    = FastQueue(5, 'cpu')

        for item in sequence:
            queue.push(item)

        self._compare_fast_queue_to_sequence(queue, sequence[5:])

    def test_queue_wrap_and_diff_lenghts(self):
        sequence = [ torch.randn((i, 2, 3, 4)) for i in range(10) ]
        queue    = Queue(5)

        for item in sequence:
            queue.push(item)

        self._compare_queue_to_sequence(queue, sequence[5:])

    def test_fast_queue_wrap_and_diff_lenghts(self):
        sequence   = [ torch.randn((i, 2, 3, 4)) for i in range(10) ]
        queue_size = sum(len(x) for x in sequence[5:])

        queue = FastQueue(queue_size, 'cpu')

        for item in sequence:
            queue.push(item)

        self._compare_fast_queue_to_sequence(queue, sequence[5:])

    def test_queue_large_wrap_and_diff_lenghts(self):
        sequence = [ torch.randn((i, 2, 3, 4)) for i in range(100) ]
        queue    = Queue(3)

        for item in sequence:
            queue.push(item)

        self._compare_queue_to_sequence(queue, sequence[-3:])

    def test_large_queue_large_wrap_and_diff_lenghts(self):
        sequence = [ torch.randn((i, 2, 3, 4)) for i in range(100) ]
        queue    = FastQueue(3, 'cpu')

        for item in sequence:
            queue.push(item)

        self._compare_fast_queue_to_sequence(queue, (sequence[-1][-3:,...],))

if __name__ == '__main__':
    unittest.main()

