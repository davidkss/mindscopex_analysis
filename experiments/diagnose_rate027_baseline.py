"""Opt-in, unsteered rate_027 diagnostic; --self-test never imports ML libraries.

--run-model loads the production model and runs >=5 completions. No SAE is
loaded and no forward/steering hooks are installed. Existing outputs are refused.
Raw logits require Transformers generate(output_logits=True) support.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import platform
import sys
import warnings
from collections import Counter
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "hagendorff_crt_rate_027"
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
KERNEL_NAMES = ("causal_conv1d", "gated_delta", "is_fast_path_available")


def candidate_layout(sequences):
    if len(sequences) != 2 or any(not s for s in sequences) or sequences[0] == sequences[1]:
        raise ValueError("Expected two distinct, nonempty token sequences")
    position = 0
    while position < min(map(len, sequences)) and sequences[0][position] == sequences[1][position]:
        position += 1
    return dict(
        common_prefix_token_ids=sequences[0][:position],
        has_common_prefix=position > 0,
        first_divergence_position_zero_based=position,
        divergence_involves_candidate_end=position == min(map(len, sequences)),
    )


def allowed_at(prefix, sequences, eos):
    allowed = set()
    for ids in sequences:
        if prefix == ids[: len(prefix)]:
            allowed.update([ids[len(prefix)]] if len(prefix) < len(ids) else eos)
    if not allowed:
        raise ValueError(f"Prefix outside candidate trie: {prefix}")
    return sorted(allowed)


class RecordingModel:
    """Delegate the original helper, changing only generation return diagnostics."""

    def __init__(self, model):
        self.model = model
        self.output = None
        self.allowed_calls = []
        self.kwargs = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, **kwargs):
        self.kwargs = dict(kwargs)
        original = kwargs["prefix_allowed_tokens_fn"]

        def record(batch_id, sequence):
            allowed = original(batch_id, sequence)
            self.allowed_calls.append(
                dict(
                    batch_id=int(batch_id),
                    sequence=sequence.tolist(),
                    allowed_token_ids=list(allowed),
                )
            )
            return allowed

        kwargs = dict(
            kwargs,
            prefix_allowed_tokens_fn=record,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
        self.output = self.model.generate(**kwargs)
        if getattr(self.output, "logits", None) is None:
            raise RuntimeError(
                "Transformers did not return raw logits; no processed-score substitute is used"
            )
        return self.output.sequences


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def kernel_inventory():
    result = []
    for name, module in list(sys.modules.items()):
        if module is None or not (
            name.startswith(("transformers.models.qwen", "causal_conv1d", "fla"))
        ):
            continue
        symbols = {}
        for key, value in vars(module).items():
            if not any(part in key for part in KERNEL_NAMES):
                continue
            if value is None or isinstance(value, (bool, str, int, float)):
                symbols[key] = value
            elif callable(value):
                symbols[key] = dict(
                    module=getattr(value, "__module__", None),
                    qualname=getattr(value, "__qualname__", None),
                )
        if symbols:
            result.append(
                dict(module=name, file=getattr(module, "__file__", None), symbols=symbols)
            )
    return result


class LogCapture(logging.Handler):
    """Capture actual Transformers messages without changing its logging level."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(
            dict(logger=record.name, level=record.levelname, message=record.getMessage())
        )

    def __enter__(self):
        logging.getLogger("transformers").addHandler(self)
        return self

    def __exit__(self, *args):
        logging.getLogger("transformers").removeHandler(self)
        self.close()


class KernelTrace:
    """Observe Python call frames, never replace a kernel or install model hooks."""

    def __init__(self):
        self.calls = Counter()
        self.previous = None

    def profile(self, frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            if any(part in name for part in KERNEL_NAMES):
                self.calls[
                    (frame.f_globals.get("__name__", ""), name, frame.f_code.co_filename)
                ] += 1

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError("An existing Python profiler is active; refusing to replace it")
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *args):
        sys.setprofile(self.previous)

    def rows(self):
        return [
            dict(module=m, function=n, file=f, calls=count)
            for (m, n, f), count in sorted(self.calls.items())
        ]


def environment(model, processor, tokenizer, torch, transformers, requested_dtype):
    packages = {}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "huggingface-hub",
        "causal-conv1d",
        "flash-linear-attention",
        "flash-attn",
        "triton",
        "nvidia-cudnn-cu12",
    ):
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = dict(version=dist.version)
            direct = dist.read_text("direct_url.json")
            if direct:
                # Avoid storing arbitrary authenticated installation URLs.
                metadata = json.loads(direct)
                packages[name]["vcs_info"] = metadata.get("vcs_info")
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    parameter_groups = Counter()
    examples = []
    for name, parameter in model.named_parameters():
        parameter_groups[(str(parameter.dtype), str(parameter.device))] += parameter.numel()
        if len(examples) < 8:
            examples.append(
                dict(name=name, dtype=str(parameter.dtype), device=str(parameter.device))
            )
    configs = {}
    for name, config in (
        ("model", model.config),
        ("text", getattr(model.config, "text_config", None)),
    ):
        if config is not None:
            configs[name] = {
                key: getattr(config, key, None)
                for key in (
                    "model_type",
                    "dtype",
                    "torch_dtype",
                    "_name_or_path",
                    "_commit_hash",
                    "_attn_implementation",
                    "_attn_implementation_internal",
                )
            }
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    revisions = {
        key: init_kwargs.get(key)
        for key in (
            "revision",
            "_commit_hash",
            "name_or_path",
            "tokenizer_file",
            "vocab_file",
            "merges_file",
        )
    }
    snapshot = None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(MODEL_ID, "config.json")
        snapshot = cached if isinstance(cached, str) else None
    except Exception as exc:
        snapshot = f"Unavailable: {type(exc).__name__}"
    hooks = {
        name: len(module._forward_hooks)
        for name, module in model.named_modules()
        if getattr(module, "_forward_hooks", None)
    }
    return dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        packages=packages,
        cuda_runtime=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        cudnn_version=torch.backends.cudnn.version(),
        gpus=[
            dict(
                index=i,
                name=torch.cuda.get_device_name(i),
                capability=list(torch.cuda.get_device_capability(i)),
            )
            for i in range(torch.cuda.device_count())
        ],
        model_id=MODEL_ID,
        requested_dtype=requested_dtype,
        model_configs=configs,
        parameter_groups=[
            dict(dtype=d, device=v, numel=n) for (d, v), n in parameter_groups.items()
        ],
        parameter_examples=examples,
        hf_device_map=getattr(model, "hf_device_map", None),
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        processor_class=f"{type(processor).__module__}.{type(processor).__qualname__}",
        tokenizer_class=f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", None),
        tokenizer_revision_metadata=revisions,
        cached_config_snapshot_hint=snapshot,
        snapshot_note="Cache hint is not proof of the exact weights/tokenizer revision loaded.",
        generation_config=model.generation_config.to_dict(),
        model_training=model.training,
        existing_forward_hooks=hooks,
        diagnostic_registered_hooks=0,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        environment_variables={
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "PYTHONHASHSEED",
            )
        },
    )


def summarize_repeats(repeats):
    branches = [r["branch_steps"] for r in repeats]
    raw = [
        [
            (
                s["step_zero_based"],
                s["prefix_token_ids"],
                s["allowed_token_ids"],
                [t["raw_logit"] for t in s["tokens"]],
            )
            for s in b
        ]
        for b in branches
    ]
    return dict(
        n_repeats=len(repeats),
        answers_identical=len({r["answer"] for r in repeats}) == 1,
        labels_identical=len({r["label"] for r in repeats}) == 1,
        generated_token_ids_identical=all(
            r["generated_token_ids"] == repeats[0]["generated_token_ids"] for r in repeats
        ),
        branch_raw_logits_exactly_identical=all(x == raw[0] for x in raw),
        comparison="Exact equality of recorded raw logits; no rounding or tolerance applied",
    )


def run(args):
    import torch
    import transformers

    from mindscopex_analysis.generation import classify_lure_answer
    from mindscopex_analysis.lure_datasets import lure_dataset_cases
    from mindscopex_analysis.models import load_qwen_text_generation_model, recommended_dtype_name
    from mindscopex_analysis.prompts import instruct_lure_case
    from mindscopex_analysis.research import _binary_choice_completion

    case = instruct_lure_case(
        next(c for c in lure_dataset_cases("hagendorff_crt") if c.case_id == CASE_ID)
    )
    dtype = recommended_dtype_name() if args.dtype == "auto" else args.dtype
    model, processor = load_qwen_text_generation_model(
        MODEL_ID, device_map=args.device_map, dtype=dtype
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    # Do not silently alter inherited decoding settings to simplify interpretation.
    gc = model.generation_config
    if getattr(gc, "num_beams", 1) not in (None, 1) or getattr(
        gc, "num_return_sequences", 1
    ) not in (None, 1):
        raise ValueError(
            "Diagnostic requires num_beams and num_return_sequences in (None, 1); config unchanged"
        )
    texts = [case.correct_answer, case.lure_answer]
    ids = [list(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
    candidates = [
        dict(
            label=label,
            text=text,
            token_ids=tokens,
            token_strings=tokenizer.convert_ids_to_tokens(tokens),
            individually_decoded_tokens=[
                tokenizer.decode([t], skip_special_tokens=False) for t in tokens
            ],
        )
        for label, text, tokens in zip(("correct", "lure"), texts, ids, strict=True)
    ]
    layout = candidate_layout(ids)
    print(
        json.dumps(dict(prompt=case.prompt, candidates=candidates, layout=layout), indent=2),
        flush=True,
    )
    eos = tokenizer.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]
    report = dict(
        case_id=CASE_ID,
        created_at=datetime.now(UTC).isoformat(),
        prompt=case.prompt,
        candidates=candidates,
        candidate_layout=layout,
        environment=environment(model, processor, tokenizer, torch, transformers, args.dtype),
        production_function_sha256=hashlib.sha256(
            inspect.getsource(_binary_choice_completion).encode()
        ).hexdigest(),
        source_hashes={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/mindscopex_analysis/research.py",
                "src/mindscopex_analysis/models.py",
                "src/mindscopex_analysis/prompts.py",
                "src/mindscopex_analysis/lure_datasets.py",
                "src/mindscopex_analysis/data/hagendorff_crt.json",
            )
        },
        kernel_inventory_before=kernel_inventory(),
        repeats=[],
        instrumentation=dict(
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
            trace_python_kernel_calls=args.trace_kernels,
            note=(
                "Production decoding helper is called directly; only return instrumentation "
                "and optional Python profiling are added."
            ),
        ),
    )
    for repetition in range(args.repeats):
        wrapper = RecordingModel(model)
        trace = KernelTrace()
        with LogCapture() as logs, warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with trace if args.trace_kernels else nullcontext():
                answer = _binary_choice_completion(wrapper, processor, case, max_new_tokens=16)
        output = wrapper.output
        prompt_length = int(wrapper.kwargs["input_ids"].shape[-1])
        generated = output.sequences[0, prompt_length:].tolist()
        if len(output.logits) != len(generated) or len(output.scores) != len(generated):
            raise RuntimeError("Raw logits/scores cannot be aligned to generated tokens")
        branches = []
        for step in range(len(generated)):
            prefix = generated[:step]
            allowed = allowed_at(prefix, ids, eos)
            recorded = [
                r
                for r in wrapper.allowed_calls
                if r["sequence"] == output.sequences[0, : prompt_length + step].tolist()
            ]
            if not recorded or any(r["allowed_token_ids"] != allowed for r in recorded):
                raise RuntimeError(
                    "Recorded production prefix constraint differs from diagnostic trie"
                )
            if len(allowed) < 2:
                continue
            logits = output.logits[step][0].detach().float()
            scores = output.scores[step][0].detach().float()
            log_probs = torch.log_softmax(logits, dim=-1)
            allowed_log_probs = torch.log_softmax(logits[allowed], dim=-1)
            tokens = [
                dict(
                    token_id=t,
                    token_string=tokenizer.convert_ids_to_tokens(t),
                    decoded_token=tokenizer.decode([t], skip_special_tokens=False),
                    raw_logit=number(logits[t].item()),
                    raw_vocab_log_softmax=number(log_probs[t].item()),
                    raw_allowed_log_softmax=number(allowed_log_probs[i].item()),
                    processed_generation_score=number(scores[t].item()),
                )
                for i, t in enumerate(allowed)
            ]
            next_ids = [
                [seq[step]] if step < len(seq) else eos for seq in ids if prefix == seq[:step]
            ]
            margin = None
            if len(next_ids) == 2 and all(len(x) == 1 for x in next_ids):
                margin = number((logits[next_ids[0][0]] - logits[next_ids[1][0]]).item())
            branches.append(
                dict(
                    step_zero_based=step,
                    prefix_token_ids=prefix,
                    allowed_token_ids=allowed,
                    tokens=tokens,
                    raw_top_allowed_token_id=allowed[int(logits[allowed].argmax().item())],
                    processed_top_allowed_token_id=allowed[int(scores[allowed].argmax().item())],
                    selected_token_id=generated[step],
                    selected_candidate_labels=[
                        label
                        for label, seq in zip(("correct", "lure"), ids, strict=True)
                        if generated[: step + 1] == seq[: step + 1]
                    ],
                    correct_minus_lure_raw_logit_margin=margin,
                )
            )
        record = dict(
            repetition=repetition + 1,
            answer=answer,
            label=classify_lure_answer(answer, case),
            generated_token_ids=generated,
            branch_steps=branches,
            first_branch=branches[0] if branches else None,
            python_kernel_calls=trace.rows(),
            transformers_log_messages=logs.messages,
            warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured],
        )
        report["repeats"].append(record)
        print(json.dumps(record, indent=2), flush=True)
        # Free full-vocabulary diagnostics before the next repetition.
        del output, wrapper
    report["consistency"] = summarize_repeats(report["repeats"])
    report["kernel_inventory_after"] = kernel_inventory()
    report["kernel_evidence_limits"] = (
        "Package presence and function bindings do not prove dispatch. "
        "Python call traces show observed Python functions only, "
        "not all native/CUDA kernels. "
        "No optimized/reference classification is inferred from missing warnings. "
        "Transformers logger warnings "
        "are captured separately from Python warnings when emitted through its logger."
    )
    return report


def self_test():
    """Exercise the actual production helper's AST with lightweight fake objects."""
    source = (ROOT / "src/mindscopex_analysis/research.py").read_text()
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_binary_choice_completion"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            fn,
        ],
        type_ignores=[],
    )

    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (1, len(values))

        def to(self, device):
            return self

        def tolist(self):
            return self.values

        def __getitem__(self, key):
            return Tensor(self.values[key[1] if isinstance(key, tuple) else key])

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = None

        def encode(self, answer, add_special_tokens):
            assert add_special_tokens is False
            return {" 80 hours": [1, 8, 3], " 20 hours": [1, 2, 3]}[answer]

        def __call__(self, prompt, return_tensors):
            return {"input_ids": Tensor([70, 71]), "attention_mask": Tensor([1, 1])}

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False and kwargs["renormalize_logits"] is True
            assert kwargs["max_new_tokens"] == 16 and kwargs["pad_token_id"] == 99
            seq = [70, 71]
            for expected, chosen in [([1], 1), ([2, 8], 8), ([3], 3), ([99], 99)]:
                assert kwargs["prefix_allowed_tokens_fn"](0, Tensor(seq)) == expected
                seq = [*seq, chosen]
            result = Tensor(seq)
            if kwargs.get("return_dict_in_generate"):
                assert kwargs["output_logits"] and kwargs["output_scores"]
                return SimpleNamespace(sequences=result, logits=[1], scores=[1])
            return result

    namespace = {"torch": SimpleNamespace(inference_mode=nullcontext)}
    exec(compile(ast.fix_missing_locations(module), "<production-helper>", "exec"), namespace)
    helper = namespace["_binary_choice_completion"]
    case = SimpleNamespace(
        correct_answer=" 80 hours",
        lure_answer=" 20 hours",
        prompt="question\nAnswer:",
        case_id=CASE_ID,
    )
    wrapper = RecordingModel(Model())
    assert (
        helper(Model(), Tokenizer(), case, max_new_tokens=16)
        == helper(wrapper, Tokenizer(), case, max_new_tokens=16)
        == "80 hours"
    )
    assert len(wrapper.allowed_calls) == 4
    assert candidate_layout([[1, 8, 3], [1, 2, 3]])["first_divergence_position_zero_based"] == 1
    assert candidate_layout([[8], [2]])["has_common_prefix"] is False
    assert allowed_at([1], [[1], [1, 2]], [99]) == [2, 99]
    try:
        allowed_at([9], [[1], [2]], [99])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid prefix accepted")
    identical = dict(
        answer="80 hours",
        label="correct",
        generated_token_ids=[1, 8, 3],
        branch_steps=[
            dict(
                step_zero_based=1,
                prefix_token_ids=[1],
                allowed_token_ids=[2, 8],
                tokens=[dict(raw_logit=1.0), dict(raw_logit=2.0)],
            )
        ],
    )
    assert summarize_repeats([identical] * 5)["branch_raw_logits_exactly_identical"]
    changed = dict(
        identical,
        branch_steps=[
            dict(identical["branch_steps"][0], tokens=[dict(raw_logit=1.0), dict(raw_logit=2.01)])
        ],
    )
    assert not summarize_repeats([identical, changed])["branch_raw_logits_exactly_identical"]
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    print(
        "PASS: production helper, proxy equivalence, prefix constraints and repeat comparisons; "
        "no ML imports or model execution."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument(
        "--run-model", action="store_true", help="Explicitly opt in to model loading/inference"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--trace-kernels",
        action="store_true",
        help="Observe Python kernel calls; adds profiling overhead",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/diagnostics/rate027_baseline_diagnostic.json"
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.repeats < 5:
        parser.error("--repeats must be at least 5")
    output = args.output.resolve()
    if output.is_relative_to(ROOT / "results/runs") or output.exists():
        parser.error(
            "Refusing results/runs or an existing output file; choose a new diagnostic path"
        )
    sys.path.insert(0, str(ROOT / "src"))
    report = run(args)
    payload = json.dumps(report, indent=2, default=str, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps(report["consistency"], indent=2))
    print(f"Diagnostic saved: {output}")


if __name__ == "__main__":
    main()
