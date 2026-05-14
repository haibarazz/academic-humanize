# Academic Humanize Binary-24 Judge Prompt

This prompt evaluates whether a model has rewritten an AI-like academic draft into
natural, faithful academic English. It is designed as a binary error-detection
rubric rather than a subjective 1-5 score rubric.

The judge receives three texts:

- INPUT: the original AI-like draft.
- OUTPUT: the model rewrite to evaluate.
- REFERENCE: the human-written academic paragraph.

Use INPUT to judge edit value and residual AI patterns. Use REFERENCE to judge
semantic fidelity, factual preservation, terminology, and academic voice.

All dimensions use the same scoring rule:

- 1 = pass, no meaningful issue found.
- 0 = issue found.

If a dimension is not applicable to this passage, score 1. Do not penalize a
passage for lacking a feature that the passage type does not require.

Return valid JSON only.

## Block M: Meaning and Factual Safety

These dimensions are hard gates. A rewrite with meaning or fact errors should not
be considered high quality even if it sounds natural.

### m1_meaning_preserved
Score 0 if OUTPUT changes, reverses, weakens, or distorts the central meaning of
REFERENCE. Score 1 if the central meaning is preserved.

### m2_key_claims_preserved
Score 0 if OUTPUT omits a key claim, result, qualification, limitation, or
important detail from REFERENCE. Score 1 if key claims and qualifications are
preserved.

### m3_no_unsupported_addition
Score 0 if OUTPUT adds a new claim, implication, causal explanation, comparison,
or conclusion not supported by INPUT or REFERENCE. Score 1 if no unsupported
substantive claim is added.

### m4_logic_relations_preserved
Score 0 if OUTPUT changes causal, contrastive, conditional, temporal, or scope
relations, such as turning correlation into causation or a limited result into a
general conclusion. Score 1 if logical relations are preserved.

### m5_numbers_citations_entities_preserved
Score 0 if OUTPUT changes, removes, invents, or corrupts numbers, years,
statistics, citations, named entities, variables, labels, or references. Score 1
if these details are preserved, or if none are present.

### m6_terms_methods_preserved
Score 0 if OUTPUT changes, removes, mistranscribes, or over-generalizes domain
terms, acronyms, model names, method names, dataset names, constructs, or
technical phrases. Score 1 if terminology is accurate and specific.

## Block V: AI Vocabulary and Template Language

These dimensions capture lexical traces commonly associated with AI-generated or
AI-assisted academic writing, including patterns discussed in Wikipedia:Signs of
AI writing and corpus studies of AI-era academic prose.

### v1_tier1_ai_words_absent
Tier-1 signals include: delve, underscore, meticulous, pivotal, intricate, realm,
showcasing, commendable, noteworthy, vibrant, tapestry, palpable, camaraderie,
prowess.

Score 0 if any Tier-1 signal appears in OUTPUT in a formulaic or ornamental way.
Score 1 if absent, or if a word appears only as necessary domain content.

### v2_tier2_ai_words_not_dense
Tier-2 signals include: holistic, multifaceted, leverage, transformative,
bolster, foster, harness, groundbreaking, cutting-edge, unwavering, invaluable,
testament, embark, pioneer, robust, comprehensive, enhance, facilitate,
innovative, significant, crucial.

Score 0 if 3 or more Tier-2 words appear, or if 2 appear in generic praise-like
use. Score 1 if use is sparse and specific.

### v3_template_phrases_absent
Template phrases include: "It is worth noting that", "It is important to note
that", "It should be noted that", "In the realm of", "In light of these
findings", "Taken together", "paves the way for", "sheds light on", "bridges the
gap", "lays the groundwork", "is a testament to", "underscores the importance".

Score 0 if any phrase appears in OUTPUT. Score 1 if absent.

### v4_promotional_puffery_absent
Score 0 if OUTPUT uses inflated or promotional academic phrasing such as
"groundbreaking", "transformative", "promising directions", "considerable
progress", "broader impact", or "important implications" without specific
evidence. Score 1 if claims are concrete and calibrated.

## Block S: Sentence and Structure Patterns

### s1_participial_tail_not_overused
Flag formulaic main-clause-plus-tail patterns such as ", demonstrating...",
", highlighting...", ", enabling...", ", suggesting...", ", reflecting...", or
", underscoring..." when used as generic explanation.

Score 0 if OUTPUT has 2 or more formulaic participial tails, or one tail that
clearly preserves an AI-like sentence from INPUT. Score 1 otherwise.

### s2_nominalization_not_overused
Flag abstract noun constructions where a verb would be more natural: "the
utilization of", "the implementation of", "the examination of", "the analysis
of", "the development of", "the advancement of", "the improvement of".

Score 0 if OUTPUT has 3 or more unnecessary nominalizations. Score 1 otherwise.

### s3_negative_parallelism_absent
Negative parallelism includes: "not only X, but also Y", "not X, but Y", "not
merely X; it is Y", "not simply X; rather, Y".

Score 0 if any such structure appears in OUTPUT. Score 1 if absent.

### s4_rule_of_three_or_adjective_stack_absent
Score 0 if OUTPUT uses content-light rhetorical tripling or stacked adjectives,
such as "robust, comprehensive, and innovative", "complex, dynamic, and
multifaceted", or "rigorous and systematic methodology". Score 1 if lists and
adjectives carry distinct technical information.

### s5_elegant_variation_absent
Score 0 if OUTPUT cycles through synonyms for the same referent in adjacent
sentences, such as "algorithm" -> "model" -> "system" -> "framework", where a
human would repeat the term or use a pronoun. Score 1 if referents are stable.

## Block D: Discourse and Academic Voice

### d1_hedging_calibrated
Score 0 if all claims are hedged at the same generic level with "may", "might",
"could potentially", or "appears to", regardless of evidence strength. Score 1
if hedging matches the evidence and claim strength.

### d2_vague_attribution_absent
Vague attribution includes: "Researchers have found", "Studies suggest", "It
has been shown", "Experts argue", "Evidence indicates", "The literature
demonstrates" without a specific citation, named study, or author.

Score 0 if OUTPUT contains 2 or more vague attributions, or one vague attribution
that supports a key claim. Score 1 otherwise.

### d3_generic_conclusion_absent
Score 0 if OUTPUT ends with a generic conclusion such as "Taken together, this
underscores...", "These findings highlight the importance...", "This provides
valuable insights...", or broad future-prospect language not present in
REFERENCE. Score 1 if the ending is specific and evidence-bound.

### d4_density_and_concision_preserved
Score 0 if OUTPUT is noticeably more verbose than REFERENCE because of redundant
phrases, filler transitions, repeated abstractions, or generic explanatory
clauses. Score 1 if wording is concise while preserving meaning.

### d5_academic_register_natural
Score 0 if OUTPUT sounds like polished marketing copy, a textbook summary, or a
generic LLM article rather than discipline-appropriate academic prose. Score 1 if
register and collocations fit the academic domain.

## Block E: Editing Quality and Formatting

### e1_substantive_edit_value
Score 0 if OUTPUT only makes superficial synonym swaps and leaves INPUT's main
AI-like structure intact. Score 1 if OUTPUT makes a meaningful improvement in
naturalness, precision, or academic voice.

### e2_not_overedited
Score 0 if OUTPUT rewrites too freely, expands scope, compresses away important
content, or becomes less aligned with REFERENCE in the name of sounding natural.
Score 1 if edit strength is appropriate.

### e3_formulaic_transition_absent
Score 0 if OUTPUT begins a sentence or paragraph with formulaic transition words
such as Moreover, Furthermore, Additionally, However, Consequently, or
Subsequently when the transition is unnecessary or copied from INPUT. Score 1 if
transitions are natural.

### e4_formatting_artifacts_absent
Score 0 if OUTPUT contains markdown artifacts, bullet-like structure, excessive
em dashes, boldface, title-case headings, quotation artifacts, or instruction
leakage. Score 1 if formatting is clean prose.

## Score Aggregation

Total score is the sum of all 24 dimensions.

Block scores:

- M_meaning_safety: m1+m2+m3+m4+m5+m6, max 6.
- V_vocabulary: v1+v2+v3+v4, max 4.
- S_structure: s1+s2+s3+s4+s5, max 5.
- D_discourse: d1+d2+d3+d4+d5, max 5.
- E_editing_format: e1+e2+e3+e4, max 4.

Interpretation:

- 22-24: strong rewrite, semantically safe and mostly free of AI traces.
- 18-21: usable rewrite with minor residual issues.
- 13-17: partial improvement with clear remaining issues.
- 0-12: weak rewrite or unsafe rewrite.

If any M dimension is 0, set hard_fail to true. A hard-fail output should be
considered unsafe for preference training even if total score is high.

## Few-Shot Examples

### Example 1: strong rewrite

INPUT:
It is worth noting that this paper delves into the intricate mechanisms linking
platform governance and seller participation, underscoring the pivotal role of
transparent rules in the realm of digital marketplaces. Taken together, these
findings highlight the importance of robust governance for fostering trust.

OUTPUT:
This paper examines how platform governance shapes seller participation in
digital marketplaces. The results show that transparent rules were associated
with higher seller engagement, particularly when enforcement procedures were
visible to participants. These findings suggest that governance design affects
trust through rule clarity and enforcement transparency.

REFERENCE:
We examined how platform governance shaped seller participation in digital
marketplaces. Transparent rules were associated with higher seller engagement,
particularly when enforcement procedures were visible to participants. The
results suggest that governance design affects trust through rule clarity and
enforcement transparency.

JSON:
{
  "m1_meaning_preserved": 1,
  "m2_key_claims_preserved": 1,
  "m3_no_unsupported_addition": 1,
  "m4_logic_relations_preserved": 1,
  "m5_numbers_citations_entities_preserved": 1,
  "m6_terms_methods_preserved": 1,
  "v1_tier1_ai_words_absent": 1,
  "v2_tier2_ai_words_not_dense": 1,
  "v3_template_phrases_absent": 1,
  "v4_promotional_puffery_absent": 1,
  "s1_participial_tail_not_overused": 1,
  "s2_nominalization_not_overused": 1,
  "s3_negative_parallelism_absent": 1,
  "s4_rule_of_three_or_adjective_stack_absent": 1,
  "s5_elegant_variation_absent": 1,
  "d1_hedging_calibrated": 1,
  "d2_vague_attribution_absent": 1,
  "d3_generic_conclusion_absent": 1,
  "d4_density_and_concision_preserved": 1,
  "d5_academic_register_natural": 1,
  "e1_substantive_edit_value": 1,
  "e2_not_overedited": 1,
  "e3_formulaic_transition_absent": 1,
  "e4_formatting_artifacts_absent": 1,
  "total": 24,
  "block_scores": {
    "M_meaning_safety": 6,
    "V_vocabulary": 4,
    "S_structure": 5,
    "D_discourse": 5,
    "E_editing_format": 4
  },
  "hard_fail": false,
  "rationale": "The rewrite preserves the governance, seller participation, transparency, engagement, and trust logic from the reference. It removes Tier-1 AI markers such as 'delves', 'intricate', 'underscoring', 'pivotal', and 'realm'. The ending remains specific to rule clarity and enforcement transparency rather than adding a generic broader-impact conclusion."
}

### Example 2: fluent but semantically unsafe

INPUT:
The proposed approach leverages a comprehensive framework to enhance prediction
accuracy, underscoring the pivotal role of LSTM models in robust demand
forecasting.

OUTPUT:
The proposed approach combines several machine learning models and demonstrates
that transformer-based forecasting is more accurate than prior LSTM baselines.

REFERENCE:
The proposed approach used an LSTM model to improve demand forecasting accuracy.
The study compared the LSTM model with traditional statistical baselines.

JSON:
{
  "m1_meaning_preserved": 0,
  "m2_key_claims_preserved": 0,
  "m3_no_unsupported_addition": 0,
  "m4_logic_relations_preserved": 0,
  "m5_numbers_citations_entities_preserved": 1,
  "m6_terms_methods_preserved": 0,
  "v1_tier1_ai_words_absent": 1,
  "v2_tier2_ai_words_not_dense": 1,
  "v3_template_phrases_absent": 1,
  "v4_promotional_puffery_absent": 1,
  "s1_participial_tail_not_overused": 1,
  "s2_nominalization_not_overused": 1,
  "s3_negative_parallelism_absent": 1,
  "s4_rule_of_three_or_adjective_stack_absent": 1,
  "s5_elegant_variation_absent": 1,
  "d1_hedging_calibrated": 1,
  "d2_vague_attribution_absent": 1,
  "d3_generic_conclusion_absent": 1,
  "d4_density_and_concision_preserved": 1,
  "d5_academic_register_natural": 1,
  "e1_substantive_edit_value": 1,
  "e2_not_overedited": 0,
  "e3_formulaic_transition_absent": 1,
  "e4_formatting_artifacts_absent": 1,
  "total": 18,
  "block_scores": {
    "M_meaning_safety": 1,
    "V_vocabulary": 4,
    "S_structure": 5,
    "D_discourse": 5,
    "E_editing_format": 3
  },
  "hard_fail": true,
  "rationale": "The output sounds cleaner but is semantically unsafe. It changes the method from LSTM to transformer-based forecasting, invents a superiority claim, and omits the comparison with traditional statistical baselines. This is a hard fail even though surface AI markers were removed."
}

### Example 3: surface cleanup with residual AI discourse

INPUT:
Moreover, the intricate relationship between socioeconomic factors and health
outcomes underscores the multifaceted nature of these disparities, highlighting
the need for a comprehensive and holistic approach to public health policy.

OUTPUT:
Furthermore, the relationship between socioeconomic factors and health outcomes
reflects the complex nature of these disparities, highlighting the need for a
more comprehensive approach to public health policy. Taken together, these
findings highlight the importance of effective policy interventions.

REFERENCE:
Socioeconomic factors interact with health outcomes in ways that complicate
policy design. Addressing health disparities requires structural intervention in
resource allocation, because incremental changes have not reduced inequities.

JSON:
{
  "m1_meaning_preserved": 1,
  "m2_key_claims_preserved": 0,
  "m3_no_unsupported_addition": 0,
  "m4_logic_relations_preserved": 0,
  "m5_numbers_citations_entities_preserved": 1,
  "m6_terms_methods_preserved": 1,
  "v1_tier1_ai_words_absent": 1,
  "v2_tier2_ai_words_not_dense": 0,
  "v3_template_phrases_absent": 0,
  "v4_promotional_puffery_absent": 0,
  "s1_participial_tail_not_overused": 0,
  "s2_nominalization_not_overused": 1,
  "s3_negative_parallelism_absent": 1,
  "s4_rule_of_three_or_adjective_stack_absent": 1,
  "s5_elegant_variation_absent": 1,
  "d1_hedging_calibrated": 1,
  "d2_vague_attribution_absent": 1,
  "d3_generic_conclusion_absent": 0,
  "d4_density_and_concision_preserved": 0,
  "d5_academic_register_natural": 0,
  "e1_substantive_edit_value": 0,
  "e2_not_overedited": 0,
  "e3_formulaic_transition_absent": 0,
  "e4_formatting_artifacts_absent": 1,
  "total": 12,
  "block_scores": {
    "M_meaning_safety": 3,
    "V_vocabulary": 1,
    "S_structure": 4,
    "D_discourse": 3,
    "E_editing_format": 1
  },
  "hard_fail": true,
  "rationale": "The output keeps the broad topic but omits the reference's key claim about structural intervention and incremental changes. It retains formulaic AI patterns such as 'Furthermore', 'highlighting the need', 'comprehensive', and 'Taken together'. The edit is mostly surface-level and adds a generic conclusion not grounded in the reference."
}

## Output Format

Respond only with one valid JSON object using exactly these keys:

{
  "m1_meaning_preserved": 0 or 1,
  "m2_key_claims_preserved": 0 or 1,
  "m3_no_unsupported_addition": 0 or 1,
  "m4_logic_relations_preserved": 0 or 1,
  "m5_numbers_citations_entities_preserved": 0 or 1,
  "m6_terms_methods_preserved": 0 or 1,
  "v1_tier1_ai_words_absent": 0 or 1,
  "v2_tier2_ai_words_not_dense": 0 or 1,
  "v3_template_phrases_absent": 0 or 1,
  "v4_promotional_puffery_absent": 0 or 1,
  "s1_participial_tail_not_overused": 0 or 1,
  "s2_nominalization_not_overused": 0 or 1,
  "s3_negative_parallelism_absent": 0 or 1,
  "s4_rule_of_three_or_adjective_stack_absent": 0 or 1,
  "s5_elegant_variation_absent": 0 or 1,
  "d1_hedging_calibrated": 0 or 1,
  "d2_vague_attribution_absent": 0 or 1,
  "d3_generic_conclusion_absent": 0 or 1,
  "d4_density_and_concision_preserved": 0 or 1,
  "d5_academic_register_natural": 0 or 1,
  "e1_substantive_edit_value": 0 or 1,
  "e2_not_overedited": 0 or 1,
  "e3_formulaic_transition_absent": 0 or 1,
  "e4_formatting_artifacts_absent": 0 or 1,
  "total": 0-24,
  "block_scores": {
    "M_meaning_safety": 0-6,
    "V_vocabulary": 0-4,
    "S_structure": 0-5,
    "D_discourse": 0-5,
    "E_editing_format": 0-4
  },
  "hard_fail": true or false,
  "rationale": "3-6 concise sentences citing specific words, omissions, additions, or patterns."
}
