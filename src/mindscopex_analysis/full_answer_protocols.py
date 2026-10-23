"""Explicit full-answer input protocols and canonical token-ID fingerprints."""
from __future__ import annotations

import hashlib
import json

DIAGNOSTIC = "diagnostic_compatible"
BEHAVIOR = "behavior_compatible"
PROTOCOLS = (DIAGNOSTIC, BEHAVIOR)


def require_protocol(value):
    if value not in PROTOCOLS:
        raise ValueError(f"input_protocol is required and must be one of {PROTOCOLS}; got {value!r}")
    return value


def token_ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def candidate_texts(case, protocol):
    require_protocol(protocol)
    answers = {"correct": case.correct_answer, "lure": case.lure_answer}
    return {kind: answer.strip() if protocol == DIAGNOSTIC else answer
            for kind, answer in answers.items()}


def protocol_spec(protocol):
    require_protocol(protocol)
    if protocol == DIAGNOSTIC:
        return dict(
            prompt="chat_template", system_prompt="", enable_thinking=False,
            candidate_text="answer.strip()", add_special_tokens_for_candidates=False,
            reference="scripts/diagnose_fresh_v2_matched_branch_margin.py:first_branch/main",
        )
    return dict(
        prompt="text_tokenizer(case.prompt, return_tensors='pt')",
        candidate_text="case.correct_answer / case.lure_answer, whitespace preserved",
        add_special_tokens_for_candidates=False, letter_candidates=False,
        reference="src/mindscopex_analysis/research.py:_binary_choice_completion",
        reference_selection="Two-candidate prefix trie; greedy; EOS allowed after a complete candidate",
    )


def capture_behavior_inputs(tokenizer, case):
    """Run the real reference with a CPU capture model; never execute model.forward."""
    import torch
    from mindscopex_analysis.research import _binary_choice_completion

    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    captures = []
    for candidate_index in (0, 1):
        class RecordingTokenizer:
            def __init__(self):
                self.tokenizer = self
                self.candidates = []

            def __getattr__(self, name):
                return getattr(tokenizer, name)

            def encode(self, text, *, add_special_tokens):
                ids = list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
                self.candidates.append(ids)
                return ids

            def __call__(self, *args, **kwargs):
                return tokenizer(*args, **kwargs)

        recorder = RecordingTokenizer()

        class CaptureModel:
            def parameters(self):
                return iter([torch.zeros(1, device="cpu")])

            def generate(self, **kwargs):
                if kwargs["do_sample"] is not False or kwargs["renormalize_logits"] is not True:
                    raise ValueError("Behavior reference selection rules changed")
                self.prompt = kwargs["input_ids"][0].tolist()
                sequence = kwargs["input_ids"].clone()
                ids = recorder.candidates[candidate_index]
                for token in ids + [kwargs["eos_token_id"][0]]:
                    allowed = kwargs["prefix_allowed_tokens_fn"](0, sequence[0])
                    if token not in allowed:
                        raise ValueError("Candidate path is outside behavioral reference trie")
                    sequence = torch.cat((sequence, torch.tensor([[token]], device="cpu")), dim=1)
                return sequence

        model = CaptureModel()
        _binary_choice_completion(model, recorder, case, max_new_tokens=16, letter_candidates=False)
        if len(recorder.candidates) != 2:
            raise ValueError("Behavior reference candidate construction changed")
        captures.append((model.prompt, recorder.candidates))
    if captures[0] != captures[1]:
        raise ValueError("Behavior reference inputs differ between candidate selections")
    return captures[0][0], dict(zip(("correct", "lure"), captures[0][1], strict=True))
