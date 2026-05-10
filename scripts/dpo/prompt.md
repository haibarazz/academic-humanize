# Academic Humanize DPO Rejected Candidate Prompt

This prompt is used by `scripts/dpo/build_dpo_rejected_candidates.py` to
generate controlled rejected candidates for DPO preference training.

The goal is not to create bad English. The goal is to create plausible
LLM-style academic rewrites that are semantically close to the human reference
but less natural and more formulaic.

## Difficulty Mix

Recommended DPO v1 mix:

| Difficulty | Purpose | Target ratio |
|---|---|---:|
| easy | Teach obvious anti-patterns | 20% |
| medium | Main preference-learning examples | 50% |
| hard | Fine-grained preference examples | 30% |

## Runtime Prompt Template

The script extracts only the block below and fills:

- `{difficulty}`
- `{input_text}`
- `{reference_text}`

<!-- BEGIN_RUNTIME_PROMPT -->
You are generating REJECTED responses for DPO preference training.

Task:
Rewrite the academic paragraph into a fluent but noticeably AI-like academic version.
The result should be a plausible LLM-generated rewrite, not broken English and not nonsense.

Important:
- Preserve the original meaning.
- Preserve all numbers, citations, acronyms, terminology, and logical relationships.
- Do not add new claims.
- Do not remove technical details.
- Do not mention that the text is AI-like.
- Output only valid JSON.

AI-like style means the rewrite should intentionally contain controlled signs of generic LLM writing.

Use lexical markers from this list when appropriate:
accentuate, ameliorate, amplify, ascertain, bolster, bustling, conceptualize,
consolidate, convey, culminate, decipher, demonstrate, depict, delineate,
delve, delve into, disseminate, elucidate, endeavor, engage, enumerate,
envision, enduring, exacerbate, expedite, foster, galvanize, harmonize,
hone, intricate, leverage, manifest, mediate, nuanced, obscure, perpetuate,
permeate, pivotal, profound, recapitulate, reconcile, rectify, reimagine,
scrutinize, substantiate, tailor, testament, transcend, traverse, underscore,
unveil, vibrant, taken together.

Also include AI-like structural patterns when appropriate:
- "not only X, but also Y"
- "not X, but Y"
- a sentence with an em dash
- a generic concluding sentence such as "Taken together, this underscores..."
- a participial tail such as ", demonstrating...", ", highlighting...", or ", enabling..."
- a formulaic transition such as "Furthermore", "Moreover", or "Additionally"
- abstract academic phrasing such as "the utilization of", "the implementation of", or "the advancement of"

Difficulty profile:
{difficulty}

If difficulty = easy:
- Use 4-5 AI lexical markers.
- Use 2 obvious structural patterns.
- Make the AI flavor clearly visible.

If difficulty = medium:
- Use 2-3 AI lexical markers.
- Use 1-2 structural patterns.
- Keep it plausible and fluent.

If difficulty = hard:
- Use only 1-2 AI lexical markers.
- Use 1 subtle structural pattern.
- Make it close to acceptable, but still less natural than the human reference.

Style constraints:
- Keep the paragraph academically fluent.
- Make it slightly over-polished and generic.
- Avoid making it obviously absurd.
- Keep length within 80%-130% of the human reference length when possible.

Input paragraph:
{input_text}

Human reference:
{reference_text}

Return valid JSON only:

{
  "rejected": "...",
  "lexical_markers_used": ["..."],
  "structural_patterns_used": ["..."],
  "difficulty": "{difficulty}"
}
<!-- END_RUNTIME_PROMPT -->

