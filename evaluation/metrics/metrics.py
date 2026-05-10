"""
评估指标层（共享）。

当前提供：
1) 语料级 BLEU
2) 句级 BLEU
3) chrF++
4) TER
5) BERTScore-F1
6) 风格相似度 / 风格差异度
7) Terminology retention helper
8) Length-ratio helper
9) Academic Humanize 格式污染诊断
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from math import sqrt
from typing import Any, Dict, List

import sacrebleu

try:
    from bert_score import score as bert_score_score
except Exception:
    bert_score_score = None


_TERM_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]{2,}")
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_META_PREFIX_PATTERN = re.compile(
    r"^\s*(?:here(?:'s| is)?|below is|the revised|revised version|polished version|"
    r"improved version|rewritten text|rewrite:|revision:|edited version|refined version)\b",
    re.IGNORECASE,
)
_META_SUFFIX_PATTERNS = [
    re.compile(r"\b(?:hope this helps|let me know|if you(?:'d)? like|i can also)\b", re.IGNORECASE),
    re.compile(r"\b(?:key improvements?|changes made|revision notes?|explanation|rationale)\b", re.IGNORECASE),
]
_EXPLANATION_BLOCK_PATTERN = re.compile(
    r"\b(?:key improvements?|changes made|revision notes?|explanation|rationale|why this version)\b\s*[:：]",
    re.IGNORECASE,
)
_BULLET_LINE_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+\S+")
_EN_STOPWORDS = {
    "the",
    "and",
    "that",
    "this",
    "these",
    "those",
    "with",
    "from",
    "into",
    "onto",
    "when",
    "where",
    "while",
    "which",
    "whose",
    "there",
    "their",
    "about",
    "after",
    "before",
    "among",
    "between",
    "because",
    "through",
    "using",
    "used",
    "based",
    "method",
    "methods",
    "result",
    "results",
    "paper",
    "study",
}


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """计算语料级 SacreBLEU 分数。"""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses 与 references 长度必须一致")
    if not hypotheses:
        return 0.0
    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)


def compute_sentence_bleu(hypothesis: str, reference: str) -> float:
    """计算单句 BLEU。"""
    hypothesis_text = (hypothesis or "").strip()
    reference_text = (reference or "").strip()
    if not hypothesis_text or not reference_text:
        return 0.0
    return float(sacrebleu.corpus_bleu([hypothesis_text], [[reference_text]]).score)


def compute_chrfpp(hypotheses: List[str], references: List[str]) -> float:
    """计算语料级 chrF++ 分数。"""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses 与 references 长度必须一致")
    if not hypotheses:
        return 0.0
    return float(
        sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            char_order=6,
            word_order=2,
            beta=2,
        ).score
    )


def compute_ter(hypotheses: List[str], references: List[str]) -> float:
    """计算语料级 TER，越低越好。"""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses 与 references 长度必须一致")
    if not hypotheses:
        return 0.0
    return float(sacrebleu.corpus_ter(hypotheses, [references]).score)


def compute_bertscore_f1(
    hypotheses: List[str],
    references: List[str],
    lang: str = "en",
    model_type: str | None = None,
    batch_size: int = 16,
    device: str | None = None,
    rescale_with_baseline: bool = False,
) -> float:
    """计算语料级 BERTScore-F1。"""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses 与 references 长度必须一致")
    if not hypotheses:
        return 0.0
    if bert_score_score is None:
        raise RuntimeError("未安装 bert-score，请先安装 `bert-score` 依赖。")

    _, _, f1 = bert_score_score(
        hypotheses,
        references,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )
    return float(f1.mean().item())


def compute_burstiness(text: str) -> float:
    """计算句长突发性：句长标准差 / 句长均值。"""
    raw = (text or "").strip()
    if not raw:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?。！？]+", raw) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(sentence.split()) for sentence in sentences]
    if len(lengths) < 2:
        return 0.0
    mean_len = sum(lengths) / len(lengths)
    if mean_len <= 0:
        return 0.0
    variance = sum((value - mean_len) ** 2 for value in lengths) / len(lengths)
    return float(sqrt(variance) / mean_len)


def compute_style_similarity(text_a: str, text_b: str) -> float:
    """计算文本相似度，值域 [0, 1]。"""
    left = (text_a or "").strip()
    right = (text_b or "").strip()
    if not left or not right:
        return 0.0
    return max(0.0, min(1.0, float(SequenceMatcher(None, left, right).ratio())))


def compute_style_diff(text_a: str, text_b: str) -> float:
    """计算风格差异度，定义为 1 - style_similarity。"""
    return 1.0 - compute_style_similarity(text_a, text_b)


def _has_bullet_or_list_tail(text: str) -> bool:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    if len(lines) < 2:
        return False

    bullet_start = None
    for idx, line in enumerate(lines):
        if _BULLET_LINE_PATTERN.match(line):
            bullet_start = idx
            break
    if bullet_start is None:
        return False

    prose_before = any(line.strip() and not _BULLET_LINE_PATTERN.match(line) for line in lines[:bullet_start])
    bullet_lines = sum(1 for line in lines[bullet_start:] if _BULLET_LINE_PATTERN.match(line))
    return prose_before and bullet_lines >= 1


def analyze_format_violations(text: str) -> Dict[str, Any]:
    """诊断 Academic Humanize 输出中的格式污染。"""
    raw = (text or "").strip()
    contains_cjk = bool(_CJK_PATTERN.search(raw))
    meta_prefix = bool(_META_PREFIX_PATTERN.search(raw))
    meta_suffix = any(pattern.search(raw) for pattern in _META_SUFFIX_PATTERNS)
    explanation_block = bool(_EXPLANATION_BLOCK_PATTERN.search(raw))
    bullet_or_list_tail = _has_bullet_or_list_tail(raw)
    format_violation = any(
        [
            contains_cjk,
            meta_prefix,
            meta_suffix,
            explanation_block,
            bullet_or_list_tail,
        ]
    )
    return {
        "contains_cjk": contains_cjk,
        "meta_prefix": meta_prefix,
        "meta_suffix": meta_suffix,
        "explanation_block": explanation_block,
        "bullet_or_list_tail": bullet_or_list_tail,
        "format_violation": format_violation,
    }


def extract_terms(text: str) -> List[str]:
    """提取术语候选。"""
    tokens = _TERM_TOKEN_PATTERN.findall(text or "")
    terms: List[str] = []
    seen = set()
    for token in tokens:
        normalized = token.lower().strip()
        if not normalized or normalized in _EN_STOPWORDS:
            continue
        if normalized.isdigit():
            continue
        keep = (
            len(normalized) >= 6
            or token.isupper()
            or any(ch.isdigit() for ch in token)
            or "-" in token
            or "_" in token
            or "/" in token
        )
        if not keep or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def compute_term_match_rate(prediction: str, reference: str) -> float:
    """计算 reference 术语在 prediction 中的命中比例。"""
    reference_terms = extract_terms(reference)
    if not reference_terms:
        return 1.0
    prediction_terms = set(extract_terms(prediction))
    hits = sum(1 for term in reference_terms if term in prediction_terms)
    return float(hits / max(len(reference_terms), 1))


def compute_length_ratio(prediction: str, reference: str) -> float:
    """计算长度比 len(prediction) / len(reference)。"""
    reference_text = (reference or "").strip()
    prediction_text = (prediction or "").strip()
    if not reference_text:
        return 0.0
    return float(len(prediction_text) / max(len(reference_text), 1))


__all__ = [
    "compute_bleu",
    "compute_sentence_bleu",
    "compute_chrfpp",
    "compute_ter",
    "compute_bertscore_f1",
    "compute_burstiness",
    "compute_style_similarity",
    "compute_style_diff",
    "analyze_format_violations",
    "extract_terms",
    "compute_term_match_rate",
    "compute_length_ratio",
]
