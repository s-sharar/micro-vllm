"""CUDA paged writes preserve exact values and reject invalid metadata before writes."""
import importlib
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class PagedKVWriteTests(unittest.TestCase):
    def setUp(self):
        self.write = importlib.import_module('micro_vllm.cuda_paged_cache')._write_paged_kv

    def arguments(self, dtype=torch.float32, device='cuda'):
        key = torch.randn((1, 2, 7, 17), dtype=dtype, device=device)
        value = torch.randn_like(key)
        pages = torch.full((5, 2, 4, 17), float('nan'), dtype=dtype, device=device)
        table = torch.tensor([3, 0, 4, 1], dtype=torch.int64, device=device)
        return [key, value, pages, pages.clone(), table, 2]

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def expected_pages(self, args):
        key, value, key_pages, value_pages, table, start = args
        expected = [key_pages.clone(), value_pages.clone()]
        for token in range(key.size(2)):
            page, slot = divmod(start + token, key_pages.size(2))
            for output, source in zip(expected, (key, value)):
                output[table[page].item(), :, slot, :] = source[0, :, token, :]
        return expected

    def assert_write(self, args):
        original = [tensor.clone() for tensor in args[:2]]
        expected = self.expected_pages(args)
        pointers = [tensor.data_ptr() for tensor in args[2:4]]
        self.assertIsNone(self.write(*args))
        for output, reference, pointer in zip(args[2:4], expected, pointers):
            self.assert_exact(output, reference)
            self.assertEqual(output.data_ptr(), pointer)
        for source, reference in zip(args[:2], original):
            self.assert_exact(source, reference)

    def test_shuffled_pages_boundaries_dtypes_and_unused_nan(self):
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.assert_write(self.arguments(dtype))

    def test_grid_stride_loop_and_odd_feature_dimension(self):
        key = torch.randn((1, 17, 513, 129), device='cuda')
        pages = torch.full((6, 17, 128, 129), float('nan'), device='cuda')
        args = [key, torch.randn_like(key), pages, pages.clone(),
                torch.tensor([4, 0, 5, 1, 3], device='cuda'), 127]
        self.assertGreater(key.numel(), 4096 * 256)
        self.assert_write(args)

    def test_multiple_chunks_fill_exact_capacity_and_rewrite_after_reset(self):
        args = self.arguments()
        for start, count in ((0, 3), (3, 5), (8, 8), (0, 1)):
            args[:2] = [torch.randn((1, 2, count, 17), device='cuda') for _ in range(2)]
            args[-1] = start
            self.assert_write(args)

    def test_independent_offset_stepped_and_broadcast_source_strides(self):
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                args = self.arguments(dtype)
                key = torch.randn((1, 6, 23, 17), device='cuda', dtype=dtype)[:, 1:5:2, 2:23:3]
                value = torch.randn((1, 16, 5, 17), device='cuda', dtype=dtype)[:, 1:15:2, 2:4].transpose(1, 2)
                self.assertNotEqual(key.stride(), value.stride())
                args[:2] = [key, value]
                self.assert_write(args)
                args[:2] = [key[:, :1, :1].expand(1, 2, 7, 17),
                            value[:, :1, :1].expand(1, 2, 7, 17)]
                self.assert_write(args)
                args[1] = args[0]
                self.assert_write(args)

    def test_preserves_special_value_bits(self):
        for dtype, integer in ((torch.float16, torch.int16), (torch.bfloat16, torch.int16),
                               (torch.float32, torch.int32)):
            with self.subTest(dtype=dtype):
                args = self.arguments(dtype)
                source = args[0]
                source.flatten()[:4] = torch.tensor([0., -0., float('inf'), -float('inf')], device='cuda')
                payload = 0x7fc12345 if dtype == torch.float32 else (0x7e11 if dtype == torch.float16 else 0x7fc1)
                source.view(integer).flatten()[4] = payload
                args[1] = source
                self.write(*args)
                for token in range(source.size(2)):
                    page, slot = divmod(args[-1] + token, args[2].size(2))
                    for output in args[2:4]:
                        self.assertTrue(torch.equal(output[args[4][page], :, slot].view(integer),
                                                    source[0, :, token].view(integer)))

    def test_current_nondefault_stream(self):
        importlib.import_module('micro_vllm.cuda_paged_cache')._load_extension()
        stream = torch.cuda.Stream()
        args = self.arguments()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000)
            args[0].fill_(7.)
            args[1].fill_(-9.)
            self.write(*args)
            observed = [tensor.clone() for tensor in args[2:4]]
        stream.synchronize()
        expected = self.expected_pages(args)
        for actual, reference in zip(observed, expected):
            self.assert_exact(actual, reference)

    def assert_rejected(self, args, message):
        original = [tensor.clone() for tensor in args[2:4]]
        with self.assertRaisesRegex((RuntimeError, TypeError, ValueError), message):
            self.write(*args)
        for output, reference in zip(args[2:4], original):
            self.assert_exact(output, reference)

    def test_rejects_invalid_shapes_strides_dtypes_and_capacity(self):
        cases = [
            (0, lambda x: x.squeeze(0), 'shape'),
            (1, lambda x: x[:, :, :-1], 'shape'),
            (0, lambda x: x.expand(2, -1, -1, -1), 'batch|shape'),
            (0, lambda x: x[:, :, :0], 'positive|shape'),
            (0, lambda x: torch.empty((1, 2, 7, 34), device='cuda')[..., ::2], 'feature stride'),
            (1, lambda x: x.double(), 'dtype'),
            (0, lambda x: x.requires_grad_(), 'gradients'),
            (0, lambda x: torch._neg_view(x), 'negative'),
            (0, lambda x: x.cpu(), 'CUDA'),
            (2, lambda x: x.transpose(0, 2), 'contiguous|shape'),
            (3, lambda x: x[:, :1], 'shape|contiguous'),
            (4, lambda x: x.int(), 'int64'),
            (4, lambda x: x.cpu(), 'CUDA'),
            (4, lambda x: x.reshape(2, 2), '1D|one-dimensional'),
            (4, lambda x: x[:0], 'positive|nonempty'),
            (4, lambda x: torch.arange(6, device='cuda'), 'capacity'),
            (4, lambda x: torch.arange(8, device='cuda')[::2], 'contiguous'),
            (5, lambda x: -1, 'start'),
            (5, lambda x: 10, 'capacity'),
        ]
        for index, change, message in cases:
            with self.subTest(argument=index, message=message):
                args = self.arguments()
                args[index] = change(args[index])
                self.assert_rejected(args, message)

    def test_rejects_sparse_and_unsupported_dtypes(self):
        args = self.arguments()
        args[0] = args[0].to_sparse()
        self.assert_rejected(args, 'strided')
        args = self.arguments()
        args[:4] = [tensor.to(torch.int32) for tensor in args[:4]]
        self.assert_rejected(args, 'dtype')

    def test_rejects_all_source_destination_and_destination_aliases(self):
        for source in (0, 1):
            for destination in (2, 3):
                with self.subTest(source=source, destination=destination):
                    args = self.arguments()
                    args[source] = args[destination].view(1, 2, 20, 17)[:, :, :7]
                    self.assert_rejected(args, 'alias|storage')
        args = self.arguments()
        args[3] = args[2]
        self.assert_rejected(args, 'alias|storage')
        args = self.arguments()
        shared = torch.empty(2 * args[2].numel(), device='cuda')
        args[2] = shared[:args[2].numel()].view_as(args[2])
        args[3] = shared[args[2].numel():].view_as(args[3])
        self.assert_rejected(args, 'alias|storage')

    def test_rejects_dlpack_aliases_with_independent_storage_metadata(self):
        for source in (0, 1):
            for destination in (2, 3):
                with self.subTest(source=source, destination=destination):
                    args = self.arguments()
                    view = args[destination].view(1, 2, 20, 17)[:, :, 1:8]
                    args[source] = torch.utils.dlpack.from_dlpack(view)
                    self.assertNotEqual(args[source].untyped_storage().data_ptr(),
                                        args[destination].untyped_storage().data_ptr())
                    self.assert_rejected(args, 'alias|storage')
        with self.subTest(alias='destinations'):
            args = self.arguments()
            args[3] = torch.utils.dlpack.from_dlpack(args[2])
            self.assert_rejected(args, 'alias|storage')
        with self.subTest(alias='table'):
            args = self.arguments()
            table_view = args[2].flatten()[2:10].view(torch.int64)
            args[4] = torch.utils.dlpack.from_dlpack(table_view)
            self.assert_rejected(args, 'alias|storage')

    def test_high_level_cache_matches_cpu_for_multilayer_chunks_and_reset(self):
        from micro_vllm.paged_kv_cache import PagedKVCache

        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                options = dict(num_layers=2, num_kv_heads=2, num_pages=5, page_size=4,
                               head_dim=17, dtype=dtype, page_table=[3, 0, 4, 1])
                cpu = PagedKVCache.allocate(**options, device='cpu')
                gpu = PagedKVCache.allocate(**options, device='cuda')
                for cache in (cpu, gpu):
                    for pages in (*cache._keys, *cache._values):
                        pages.fill_(float('nan'))
                buffers = (*gpu._keys, *gpu._values, gpu._device_page_table)
                pointers = [tensor.data_ptr() for tensor in buffers]
                for layer in range(2):
                    key = torch.randn((1, 4, 32, 17), device='cuda', dtype=dtype)[:, ::2, ::2]
                    value = torch.randn((1, 16, 2, 17), device='cuda', dtype=dtype).transpose(1, 2)
                    original = [tensor.clone() for tensor in (key, value)]
                    for begin, end in ((0, 3), (3, 8), (8, 16)):
                        chunks = (key[:, :, begin:end], value[:, :, begin:end])
                        self.assertIs(gpu.append(layer, *chunks), gpu)
                        cpu.append(layer, *(chunk.cpu() for chunk in chunks))
                        self.assertEqual(gpu.length(layer), end)
                        for actual, expected in zip(gpu.gather(layer), cpu.gather(layer)):
                            self.assert_exact(actual.cpu(), expected)
                    for source, saved in zip((key, value), original):
                        self.assert_exact(source, saved)
                self.assertEqual(gpu.sequence_length, 16)
                for actual, expected in zip((*gpu._keys, *gpu._values), (*cpu._keys, *cpu._values)):
                    self.assert_exact(actual.cpu(), expected)
                saved = [tensor.clone() for tensor in buffers]
                self.assertIs(gpu.reset(), gpu)
                self.assertEqual(gpu.sequence_length, 0)
                for layer in range(2):
                    self.assertEqual(gpu.gather(layer)[0].shape, (1, 2, 0, 17))
                for tensor, expected, pointer in zip(buffers, saved, pointers):
                    self.assert_exact(tensor, expected)
                    self.assertEqual(tensor.data_ptr(), pointer)
                fresh = torch.randn((1, 2, 1, 17), device='cuda', dtype=dtype)
                gpu.append(0, fresh, fresh)
                for actual in gpu.gather(0):
                    self.assert_exact(actual, fresh)
                self.assertEqual([tensor.data_ptr() for tensor in buffers], pointers)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'two CUDA GPUs required')
    def test_device_mismatch_and_current_device_guard(self):
        for index in range(5):
            args = self.arguments(device='cuda:0')
            args[index] = args[index].to('cuda:1')
            self.assert_rejected(args, 'device')
        with torch.cuda.device(1):
            self.assert_write(self.arguments(device='cuda:0'))
            self.assertEqual(torch.cuda.current_device(), 1)


if __name__ == '__main__':
    unittest.main()
