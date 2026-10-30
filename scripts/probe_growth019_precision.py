#!/usr/bin/env python3
"""CPU scalar precision/reference-selection probe; never loads an LLM or uses CUDA.

BF16 emulation is IEEE round-to-nearest-even from FP32 bit patterns. It does not
claim to reproduce Transformers or the dtype/rounding of an A100 logits tensor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.trace_fresh_v2_first_divergence import write_new


def fp32(value):
    return struct.unpack('>f', struct.pack('>f', value))[0]


def bf16_rne(value):
    if not math.isfinite(value):
        raise ValueError('Probe requires finite scalar values')
    bits = struct.unpack('>I', struct.pack('>f', value))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('>f', struct.pack('>I', rounded))[0]


def first_argmax(values):
    """Explicit first-maximum reference rule, not an inferred production rule."""
    return max(range(len(values)), key=values.__getitem__)


def full_vocab_choice(pair, allowed=(17, 20)):
    logits = [float('-inf')] * 32
    for token, value in zip((17, 20), pair, strict=True):
        if token in allowed:
            logits[token] = value
    return first_argmax(logits)


def scalar_probe():
    rows = []
    for name, pair in [('equal', (16.375, 16.375)),
                       ('correct_plus_0.01', (16.375, 16.385)),
                       ('correct_minus_0.01', (16.375, 16.365)),
                       ('correct_plus_0.125', (16.375, 16.5))]:
        original = list(map(fp32, pair))
        quantized = list(map(bf16_rne, original))
        widened = list(map(fp32, quantized))
        rows.append(dict(case=name, fp32_before_quantization=original, bf16_values=quantized,
                         fp32_after_bf16=widened, reference_fp32_token=full_vocab_choice(original),
                         reference_bf16_token=full_vocab_choice(quantized),
                         reference_widened_token=full_vocab_choice(widened)))
    return dict(mode='stdlib representation emulation and explicit reference argmax', rows=rows,
                bf16_spacing_at_16_375=bf16_rne(16.5)-bf16_rne(16.375),
                allowed_order_probe=[dict(allowed=list(order), full_vocab_reference_token=full_vocab_choice(
                    [16.375,16.375], order), gathered_first_max_token=order[first_argmax([16.375,16.375])])
                    for order in ((17,20),(20,17))],
                note='Gathered-vector first-index selection differs from vocabulary-index selection; neither proves HF dispatch.')


def torch_probe():
    if importlib.util.find_spec('torch') is None:
        return dict(status='not installed', actual_tensor_cases=0)
    import torch
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        for pair in ((16.375,16.375),(16.375,16.385),(16.375,16.5)):
            scores = torch.full((32,), -torch.inf, dtype=dtype, device='cpu')
            scores[17], scores[20] = pair
            winner, widened = int(scores.argmax().item()), int(scores.float().argmax().item())
            rows.append(dict(dtype=str(dtype), input_pair=list(pair), represented_pair=scores[[17,20]].float().tolist(),
                             argmax_token=winner, argmax_after_float=widened))
    return dict(status='CPU tensors executed', version=torch.__version__, actual_tensor_cases=len(rows), rows=rows,
                note='No model/generation or CUDA tensors; installed Torch need not match the historical A100 build.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f'Refusing existing output: {args.output}')
    result = dict(scalar=scalar_probe(), torch=torch_probe(),
                  limitations=['No Transformers generation is executed.',
                               'Widening already quantized values does not recover pre-quantization information.',
                               'Actual logits/scores dtype at selection is not stored in the historical capture.'])
    write_new(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
