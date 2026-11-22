"""Native batched paged decode and scatter: per-row tables, lengths and starts."""
import subprocess
import sys
import unittest

import torch

from micro_vllm import cuda_attention
from test.test_cuda_gqa import fp64_reference


def shared_pages(rows, page_size, heads, dimension, dtype, spare=3):
    """Lay rows' K/V into one NaN-poisoned pool with interleaved, shuffled physical IDs."""
    hkv = heads[1]
    counts = [-(-row[1].shape[2] // page_size) for row in rows]
    total = sum(counts) + spare
    physical = torch.randperm(total, generator=torch.Generator().manual_seed(11)).tolist()
    pages = torch.full((total, hkv, page_size, dimension), float('nan'), dtype=dtype, device='cuda')
    values = torch.full_like(pages, float('nan'))
    width = max(counts) + 1
    table = torch.full((len(rows), width), -1, dtype=torch.int64, device='cuda')
    row_tables, cursor = [], 0
    owners = [row for logical in range(max(counts)) for row in range(len(rows))
              if logical < counts[row]]
    assigned = {row: [] for row in range(len(rows))}
    for row in owners:
        assigned[row].append(physical[cursor])
        cursor += 1
    for row, (_, key, value) in enumerate(rows):
        ids, length = assigned[row], key.shape[2]
        for logical, page in enumerate(ids):
            start, stop = logical * page_size, min((logical + 1) * page_size, length)
            pages[page, :, :stop - start] = key[0, :, start:stop]
            values[page, :, :stop - start] = value[0, :, start:stop]
        table[row, :len(ids)] = torch.tensor(ids, device='cuda')
        row_tables.append(torch.tensor(ids, dtype=torch.int64, device='cuda'))
    return pages, values, table, row_tables


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class NativeBatchedPagedAttentionTests(unittest.TestCase):
    def setUp(self):
        self.extension = cuda_attention._load_extension()
        self.batch = self.extension._decode_attention_paged_batch
        self.single = self.extension._decode_attention_paged

    def rows(self, lengths, heads=(14, 2), dimension=64, dtype=torch.float16, seed=7):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        rows = []
        for length in lengths:
            query = torch.randn(1, heads[0], 1, dimension, dtype=dtype, device='cuda',
                                generator=generator)
            key = torch.randn(1, heads[1], length, dimension, dtype=dtype, device='cuda',
                              generator=generator)
            rows.append((query, key, torch.randn(key.shape, dtype=dtype, device='cuda',
                                                 generator=generator)))
        return rows

    def run_batch(self, rows, page_size, heads, dimension, dtype):
        pages, values, table, row_tables = shared_pages(rows, page_size, heads, dimension, dtype)
        query = torch.cat([row[0] for row in rows])
        lengths = torch.tensor([row[1].shape[2] for row in rows], dtype=torch.int64, device='cuda')
        return self.batch(query, pages, values, table, lengths), pages, values, row_tables

    def test_rows_match_single_request_decode_bit_for_bit(self):
        for dtype in (torch.float32, torch.float16):
            for page_size in (3, 16):
                for heads, dimension in (((14, 2), 64), ((6, 2), 33), ((1, 1), 257)):
                    lengths = (1, page_size, page_size + 1, 31, 33, 65, 257)
                    with self.subTest(dtype=dtype, page=page_size, heads=heads, dim=dimension):
                        rows = self.rows(lengths, heads, dimension, dtype)
                        actual, pages, values, row_tables = self.run_batch(
                            rows, page_size, heads, dimension, dtype)
                        self.assertEqual(actual.shape, (len(rows), heads[0], 1, dimension))
                        self.assertTrue(actual.is_contiguous())
                        self.assertTrue(torch.isfinite(actual).all())
                        for row, (query, key, value) in enumerate(rows):
                            single = self.single(query, pages, values, row_tables[row], key.shape[2])
                            torch.testing.assert_close(actual[row:row + 1], single, rtol=0, atol=0)
                            torch.testing.assert_close(
                                actual[row:row + 1], cuda_attention.decode_attention_gqa(query, key, value),
                                rtol=0, atol=0)
                            tolerance = (dict(rtol=2e-3, atol=2e-3) if dtype == torch.float16
                                         else dict(rtol=2e-4, atol=2e-5))
                            torch.testing.assert_close(actual[row:row + 1],
                                                       fp64_reference(query, key, value), **tolerance)

    def test_single_row_batch_is_the_single_request_case(self):
        rows = self.rows((65,))
        actual, pages, values, row_tables = self.run_batch(rows, 16, (14, 2), 64, torch.float16)
        expected = self.single(rows[0][0], pages, values, row_tables[0], 65)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_strided_query_rows_and_heads(self):
        rows = self.rows((5, 40))
        pages, values, table, _ = shared_pages(rows, 4, (14, 2), 64, torch.float16)
        storage = torch.zeros(4, 30, 2, 70, dtype=torch.float16, device='cuda')
        query = storage[::2, 1:29:2, 1:2, 3:67]
        query.copy_(torch.cat([row[0] for row in rows]))
        self.assertFalse(query.is_contiguous())
        lengths = torch.tensor([5, 40], device='cuda')
        expected = self.batch(query.contiguous(), pages, values, table, lengths)
        torch.testing.assert_close(self.batch(query, pages, values, table, lengths), expected,
                                   rtol=0, atol=0)

    def test_out_of_range_row_lengths_and_page_ids_poison_only_that_row(self):
        rows = self.rows((9, 20, 7))
        pages, values, table, _ = shared_pages(rows, 4, (14, 2), 64, torch.float16)
        query = torch.cat([row[0] for row in rows])
        good = self.batch(query, pages, values, table, torch.tensor([9, 20, 7], device='cuda'))
        width = table.shape[1]
        for bad_length in (0, -1, width * 4 + 1):
            with self.subTest(length=bad_length):
                lengths = torch.tensor([9, bad_length, 7], device='cuda')
                actual = self.batch(query, pages, values, table, lengths)
                self.assertTrue(torch.isnan(actual[1]).all())
                torch.testing.assert_close(actual[0::2], good[0::2], rtol=0, atol=0)
        actual = self.batch(query, pages, values, table, torch.tensor([9, 20, 8 + 4], device='cuda'))
        self.assertTrue(torch.isnan(actual[2]).all())
        torch.testing.assert_close(actual[:2], good[:2], rtol=0, atol=0)

    def test_rejects_invalid_batch_metadata_before_launch(self):
        rows = self.rows((9, 20))
        pages, values, table, _ = shared_pages(rows, 4, (14, 2), 64, torch.float16)
        query = torch.cat([row[0] for row in rows])
        lengths = torch.tensor([9, 20], device='cuda')
        valid = [query, pages, values, table, lengths]
        cases = [
            (0, query[:1], 'rows'), (0, query.repeat(1, 1, 2, 1), 'one query'),
            (0, query.float(), 'same dtype'), (0, query[:, :3], 'divisible'),
            (3, table[0], 'two-dimensional'), (3, table.int(), 'int64'),
            (3, table[:1], 'rows'), (3, table[:, :0].contiguous(), 'positive'),
            (3, table.t().contiguous().t(), 'contiguous'),
            (3, torch.zeros((2, pages.shape[0] + 1), dtype=torch.int64, device='cuda'),
             'physical pages'),
            (3, table.cpu(), 'CUDA'),
            (4, lengths.int(), 'int64'), (4, lengths[:1], 'rows'),
            (4, lengths.unsqueeze(0), 'one-dimensional'), (4, lengths.cpu(), 'CUDA'),
            (4, lengths.repeat_interleave(2)[::2], 'contiguous'),
        ]
        for index, invalid, message in cases:
            args = valid.copy()
            args[index] = invalid
            with self.subTest(argument=index, message=message), \
                    self.assertRaisesRegex(RuntimeError, message):
                self.batch(*args)

    def test_batched_decode_launches_one_kernel_in_clean_process(self):
        # Earlier GPU tests in the same process can lose CUPTI activities.
        script = '''
import torch
from micro_vllm import cuda_attention
extension = cuda_attention._load_extension()
pages = torch.ones(8, 2, 16, 64, device='cuda', dtype=torch.float16)
table = torch.tensor([[3, 0, -1], [5, 1, 7]], device='cuda')
lengths = torch.tensor([17, 40], device='cuda')
q = torch.zeros(2, 14, 1, 64, device='cuda', dtype=torch.float16)
extension._decode_attention_paged_batch(q, pages, pages, table, lengths)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA]) as profile:
    actual = extension._decode_attention_paged_batch(q, pages, pages, table, lengths)
    torch.cuda.synchronize()
kernels = [event.name for event in profile.events()
           if event.device_type == torch.autograd.DeviceType.CUDA]
assert len(kernels) == 1, kernels
assert 'decode_attention_online_kernel' in kernels[0], kernels
torch.testing.assert_close(actual, torch.ones_like(q))
print('BATCHED_SINGLE_KERNEL_OK')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('BATCHED_SINGLE_KERNEL_OK', result.stdout)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class NativeBatchedPagedWriteTests(unittest.TestCase):
    def setUp(self):
        from micro_vllm import cuda_paged_cache

        self.module = cuda_paged_cache
        self.write = cuda_paged_cache._write_paged_kv_batch

    def arguments(self, dtype=torch.float16):
        generator = torch.Generator(device='cuda').manual_seed(3)
        key = torch.randn((3, 2, 6, 17), dtype=dtype, device='cuda', generator=generator)
        value = torch.randn(key.shape, dtype=dtype, device='cuda', generator=generator)
        pages = torch.full((9, 2, 4, 17), float('nan'), dtype=dtype, device='cuda')
        table = torch.tensor([[3, 0, -1], [8, 2, 5], [6, -1, -1]], device='cuda')
        starts = torch.tensor([2, 5, 0], device='cuda')
        counts = torch.tensor([6, 1, 3], device='cuda')
        return [key, value, pages, pages.clone(), table, starts, counts]

    def expected(self, args):
        key, value, key_pages, value_pages, table, starts, counts = args
        expected = [key_pages.clone(), value_pages.clone()]
        for row in range(key.shape[0]):
            for token in range(counts[row].item()):
                page, slot = divmod(starts[row].item() + token, key_pages.shape[2])
                for output, source in zip(expected, (key, value)):
                    output[table[row, page].item(), :, slot] = source[row, :, token]
        return expected

    def test_rows_write_only_valid_tokens_at_their_own_positions(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                args = self.arguments(dtype)
                expected = self.expected(args)
                sources = [tensor.clone() for tensor in args[:2]]
                self.assertIsNone(self.write(*args))
                for actual, reference in zip(args[2:4], expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
                for source, reference in zip(args[:2], sources):
                    torch.testing.assert_close(source, reference, rtol=0, atol=0)

    def test_rows_match_independent_single_request_writes(self):
        args = self.arguments()
        key, value, key_pages, value_pages, table, starts, counts = args
        single_key, single_value = key_pages.clone(), value_pages.clone()
        for row in range(3):
            count = counts[row].item()
            self.module._write_paged_kv(key[row:row + 1, :, :count], value[row:row + 1, :, :count],
                                        single_key, single_value,
                                        table[row][table[row] >= 0].contiguous(),
                                        starts[row].item())
        self.write(*args)
        torch.testing.assert_close(key_pages, single_key, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(value_pages, single_value, rtol=0, atol=0, equal_nan=True)

    def test_strided_sources(self):
        args = self.arguments()
        storage = torch.randn(3, 4, 12, 20, dtype=torch.float16, device='cuda')
        args[0] = storage[:, 1:3, ::2, 2:19]
        args[1] = storage[:, 2:4, 1::2, 1:18]
        expected = self.expected(args)
        self.write(*args)
        torch.testing.assert_close(args[2], expected[0], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(args[3], expected[1], rtol=0, atol=0, equal_nan=True)

    def test_positions_outside_a_row_table_are_never_written(self):
        # Row 0 now runs past its two pages, into table padding and past the table width.
        args = self.arguments()
        args[5] = torch.tensor([10, 5, 0], device='cuda')
        before = [tensor.clone() for tensor in args[2:4]]
        self.write(*args)
        for actual, original in zip(args[2:4], before):
            touched = ~torch.isnan(actual) & torch.isnan(original)
            self.assertFalse(touched[[3, 0, 8]].any())

    def test_rejects_invalid_batch_metadata_before_writes(self):
        args = self.arguments()
        key, value, key_pages, value_pages, table, starts, counts = args
        cases = [
            (0, key[:2], 'same shape'), (1, value[:, :1], 'same shape'),
            (4, table[0], 'two-dimensional'), (4, table[:2], 'rows'), (4, table.int(), 'int64'),
            (4, torch.zeros((3, 10), dtype=torch.int64, device='cuda'), 'physical page'),
            (5, starts.int(), 'int64'), (5, starts[:2], 'rows'), (5, starts.cpu(), 'CUDA'),
            (6, counts.unsqueeze(0), 'one-dimensional'), (6, counts[:2], 'rows'),
            (2, key_pages[..., :8], 'shape'),
            (5, starts.repeat_interleave(2)[::2], 'contiguous'),
        ]
        for index, invalid, message in cases:
            current = [tensor.clone() for tensor in args[2:4]]
            changed = args.copy()
            changed[index] = invalid
            with self.subTest(argument=index, message=message), \
                    self.assertRaisesRegex(RuntimeError, message):
                self.write(*changed)
            for actual, original in zip(args[2:4], current):
                torch.testing.assert_close(actual, original, rtol=0, atol=0, equal_nan=True)
        aliased = args.copy()
        aliased[0], aliased[1] = key_pages[:3], key_pages[3:6]
        with self.assertRaisesRegex(RuntimeError, 'alias'):
            self.write(*aliased)

    def test_batched_write_launches_one_kernel_in_clean_process(self):
        script = '''
import torch
from micro_vllm import cuda_paged_cache
key = torch.ones(2, 2, 5, 64, device='cuda', dtype=torch.float16)
pages = torch.zeros(6, 2, 4, 64, device='cuda', dtype=torch.float16)
values = torch.zeros_like(pages)
table = torch.tensor([[3, 0], [5, 1]], device='cuda')
starts, counts = torch.tensor([0, 2], device='cuda'), torch.tensor([5, 1], device='cuda')
cuda_paged_cache._write_paged_kv_batch(key, key, pages, values, table, starts, counts)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA]) as profile:
    cuda_paged_cache._write_paged_kv_batch(key, key, pages, values, table, starts, counts)
    torch.cuda.synchronize()
kernels = [event.name for event in profile.events()
           if event.device_type == torch.autograd.DeviceType.CUDA]
assert len(kernels) == 1, kernels
assert 'write_paged_kv_kernel' in kernels[0], kernels
print('BATCHED_WRITE_KERNEL_OK')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('BATCHED_WRITE_KERNEL_OK', result.stdout)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class PinnedStagingLifetimeTests(unittest.TestCase):
    def test_dropped_pinned_source_is_not_reused_before_its_async_copy_completes(self):
        """PagedKVBatch relies on the pinned host allocator recording the copy's stream event."""
        torch.empty(1, device='cuda')
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            staging = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64, pin_memory=True)
            torch.cuda._sleep(200_000_000)
            device = staging.to('cuda', non_blocking=True)
        pointer = staging.data_ptr()
        del staging
        reused = torch.full((2, 3), -9, dtype=torch.int64, pin_memory=True)
        self.assertNotEqual(reused.data_ptr(), pointer)
        stream.synchronize()
        self.assertEqual(device.tolist(), [[1, 2, 3], [4, 5, 6]])


if __name__ == '__main__':
    unittest.main()
