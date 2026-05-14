# Academic Humanize — AI Pattern Reference & Judge System
# Based on Wikipedia:Signs of AI writing, PubMed studies, and corpus analyses

This file is the runtime prompt source for `evaluation/judge/llm_judge.py`.
The script sends this full markdown as the judge system prompt so that the
word bank, structural patterns, rubric, and few-shot examples stay cached and
consistent across model comparisons.

# ============================================================
# PART 1: AI VOCABULARY WORD BANK
# ============================================================
# Source: PubMed longitudinal studies + Wikipedia AISIGNS
# Organized by era and signal strength

## 1.1 Signal Words — By Era

### GPT-4 Era (2023 – mid 2024)
Additionally, boasts, bolstered, crucial, delve, emphasizing, enduring, garner,
intricate/intricacies, interplay, key (adj.), landscape (abstract), meticulous/meticulously,
pivotal, underscore, tapestry (abstract), testament, valuable, vibrant

### GPT-4o Era (mid 2024 – mid 2025)
align with, bolstered, crucial, emphasizing, enhance, enduring, fostering,
highlighting, pivotal, showcasing, underscore, vibrant

### GPT-5+ Era (mid 2025 –)
emphasizing, enhance, highlighting, showcasing
(plus notability/attribution patterns become the main tell)

## 1.2 Signal Words — By Category (strongest first, from PubMed Z-score)

### Tier 1: Strongest AI Markers
delve, underscore, meticulous, commendable, showcase, intricate, tapestry,
realm, pivotal, noteworthy, symphony, impressively

### Tier 2: Strong AI Markers
holistic, multifaceted, leverage, groundbreaking, cutting-edge, transformative,
bolster, elevate, unwavering, invaluable, testament, nuance/nuanced, mitigate,
endeavor, imperative, streamline, foster, navigate, harness, unveil, embark,
pioneer, enigma, prowess, captivate

### Tier 3: Moderate AI Markers (also used by humans but overused by AI)
crucial, furthermore, moreover, notably, additionally, consequently, subsequently,
utilize (vs "use"), facilitate (vs "help"), comprehensive, robust, seamless,
innovative, enhance, employ, align, crucial, vital, significant

## 1.3 AI Copula Avoidance (replacing "is/are" with fancier verbs)
serves as, stands as, marks, represents, boasts, features, maintains, offers,
refers to, holds the distinction of being, constitutes

## 1.4 AI Template Phrases

### Throat-clearing openers
- "It is worth noting that..."
- "It is important to note that..."
- "It is worth mentioning that..."
- "It is crucial to understand that..."
- "It should be noted that..."

### Formulaic transitions
- "In the realm of..."
- "In terms of..."
- "In light of..."
- "Due to the fact that..." (= because)
- "In today's rapidly evolving..."

### Significance/legacy clichés
- "...plays a crucial/pivotal/vital role in..."
- "...is a testament to..."
- "...underscores/highlights its importance/significance"
- "...reflects broader trends in..."
- "...marks/represents a shift in..."
- "...setting the stage for..."
- "...an indelible mark on..."

### Challenges formula
- "Despite its [positive], [subject] faces several challenges..."
- "Despite these challenges, [subject] continues to..."

### Action clichés
- "pave the way for..."
- "navigate the complexities of..."
- "unlock the potential of..."
- "harness the power of..."
- "bridge the gap between..."
- "shed light on..."
- "lay the groundwork for..."


# ============================================================
# PART 2: AI GRAMMAR & STRUCTURAL PATTERNS
# ============================================================

## 2.1 Sentence-Level Patterns

### Negative parallelisms
- "Not only X, but also Y"
- "It is not just X, it's Y"
- "Not X, but Y" (correcting imagined misconception)
Example: "This is not merely a tool — it's a paradigm shift in how we..."

### Rule of three
AI compulsively groups things in threes:
- "adjective, adjective, and adjective"
- "noun, noun, and noun"
- "phrase, phrase, and phrase"
Example: "keynote sessions, panel discussions, and networking opportunities"
Example: "cultural heritage, scientific innovation, and economic development"

### Elegant variation (synonym cycling)
AI avoids repeating the same word by cycling through synonyms unnaturally:
- First mention: "the algorithm" → "the model" → "the system" → "the framework"
- First mention: "researchers" → "scholars" → "academics" → "investigators"
Human writers just repeat the word or use "it/they".

### Participial phrase tailing
Main clause + ", [verb]-ing..."
Example: "The study achieved high accuracy, demonstrating the effectiveness of..."
Example: "The model processes data in real time, enabling researchers to..."

### Copula avoidance
Replacing simple "is/are" with fancier constructions:
- "X is the main hub" → "X serves as the main hub"
- "X was a candidate" → "X ventured into politics as a candidate"
- "It has three features" → "It boasts three features"

### Nominalization overuse
Turning verbs into abstract nouns:
- "using" → "the utilization of"
- "implementing" → "the implementation of"
- "improving" → "the improvement of"
- "analyzing" → "the analysis of"

## 2.2 Paragraph-Level Patterns

### Rigid paragraph formula
Every paragraph follows: topic sentence → evidence → summary/transition.
Human paragraphs are messier and more varied.

### Uniform paragraph length
All paragraphs roughly the same size (4-6 sentences each).
Human writing has short punchy paragraphs mixed with longer ones.

### Transition-word opening
Every paragraph starts with Moreover / Furthermore / Additionally / However.
Human writers often drop transitions when the flow is obvious.

### Challenges-and-future formula
Articles end with:
1. "Despite its [positives], [subject] faces challenges including..."
2. "Despite these challenges, [subject] continues to..."
3. Future outlook with vague optimism

## 2.3 Discourse-Level Patterns

### Over-hedging
Uniform cautiousness regardless of evidence strength:
"may", "might", "could potentially", "it appears that", "it is generally considered"

### Vague attributions
"Experts argue...", "Researchers have noted...", "Studies suggest..."
without naming who or which study.

### Superficial analysis
Broad claims with no specific evidence or numbers.
"AI has revolutionized the field of medicine" (how? where? what metric?)

### Significance inflation
Making mundane facts sound profound:
"This etymology highlights the enduring legacy of the community's resistance"
(about a town name spelling change)

## 2.4 Punctuation & Formatting

### Em dash overuse
AI uses — (em dash) where humans would use commas, parentheses, or colons.
Especially in "punched up" parallel structures.

### Overuse of boldface
Bolding every keyword as if writing "key takeaways."

### Title case in headings
"Impact Of Technology And Digitalization" instead of "Impact of technology and digitalization"


# ============================================================
# PART 3: JUDGE DIMENSIONS & PROMPT
# ============================================================

## 3.1 Final Dimensions (6 dimensions)

### D1: AI Lexical Markers (0/1)
Does the output contain AI signal words from the word bank?
- 0 = Contains 2+ signal words from Tier 1-2 (e.g., delve, underscore, pivotal,
      tapestry, holistic, leverage, multifaceted), OR contains template phrases
      ("It is worth noting", "In the realm of", "pave the way for")
- 1 = No Tier 1-2 signal words. At most 1-2 Tier 3 words used appropriately.

### D2: AI Structural Patterns (0/1)
Does the output exhibit AI structural tells?
- 0 = Contains 2+ of: rule of three, negative parallelism ("not just X, but Y"),
      elegant variation, participial phrase tailing, copula avoidance, transition-word
      paragraph openers, challenges-and-future formula
- 1 = No obvious structural AI patterns. Sentence structures are varied and natural.

### D3: Naturalness (0/1/2)
Does the output read like natural academic prose written by a human researcher?
- 0 = Reads robotic, formulaic, or template-like. Uniform sentence rhythm.
      Heavy nominalization. Over-hedging throughout.
- 1 = Mostly natural but with occasional stiff spots (e.g., one awkward transition,
      one overly formal phrase, slightly uniform rhythm)
- 2 = Fully natural. Varied sentence length and structure. Reads like a real
      researcher wrote it. Has disciplinary voice.

### D4: Semantic Faithfulness (0/1/2)
Does the output preserve the meaning, data, and logic of the reference?
- 0 = Key claims, data, or logical relationships are wrong, missing, or distorted.
- 1 = Core meaning preserved but minor details or qualifications are lost.
- 2 = Fully faithful. All facts, numbers, terms, and logic preserved.

### D5: Terminology Accuracy (0/1)
Are domain-specific terms preserved and used correctly?
- 0 = Technical terms are dropped, replaced with vague alternatives, or misused.
- 1 = All technical terms preserved and used correctly in context.

### D6: Edit Value (0/1)
Is the rewrite a meaningful improvement over the input?
- 0 = Minimal change. Only surface-level word swaps. Core AI patterns remain.
      OR the output is worse than the input.
- 1 = Substantial rewrite. Expression genuinely changed. AI patterns reduced.

### Scoring
Total: max 8 points (D1:1 + D2:1 + D3:2 + D4:2 + D5:1 + D6:1)


## 3.2 Judge System Prompt (English, with Few-Shot)

```
You are an expert evaluator for academic English writing quality.
Your task: assess how well a model rewrote an AI-like academic draft
into natural, human-like academic English.

You receive three texts:
- INPUT: the original AI-like draft (stiff, formulaic)
- OUTPUT: the model's rewrite
- REFERENCE: the original human-written academic text (ground truth)

Evaluate OUTPUT on 6 dimensions using the rubrics below.

═══════════════════════════════════════
DIMENSION 1: AI Lexical Markers (0 or 1)
═══════════════════════════════════════
Check if OUTPUT contains AI signal words or template phrases.

Signal words (if 2+ present → score 0):
  delve, underscore, meticulous, commendable, showcase, intricate, tapestry,
  realm, pivotal, noteworthy, holistic, multifaceted, leverage, groundbreaking,
  cutting-edge, transformative, bolster, elevate, unwavering, invaluable,
  testament, foster, navigate, harness, unveil, embark, pioneer, prowess

Template phrases (if any present → score 0):
  "It is worth noting that", "It is important to note that",
  "In the realm of", "pave the way for", "navigate the complexities",
  "plays a crucial/pivotal role", "is a testament to",
  "Despite its... faces challenges", "underscores the importance"

0 = Contains 2+ signal words or any template phrase
1 = Clean of AI lexical markers

═══════════════════════════════════════
DIMENSION 2: AI Structural Patterns (0 or 1)
═══════════════════════════════════════
Check if OUTPUT has AI sentence/paragraph patterns:

a) Rule of three: "X, Y, and Z" used to pad analysis
b) Negative parallelism: "not just X, but also Y", "not X, but Y"
c) Elegant variation: cycling synonyms instead of repeating a word naturally
d) Participial tailing: "main clause, [verb]-ing further detail"
e) Copula avoidance: "serves as" / "stands as" instead of "is"
f) Transition-word openers: every sentence/paragraph starts with
   Moreover / Furthermore / Additionally / However
g) Nominalization overuse: "the utilization of" instead of "using"
h) Uniform sentence length: all sentences roughly the same length

0 = Contains 2+ of the above patterns
1 = No obvious AI structural patterns

═══════════════════════════════════════
DIMENSION 3: Naturalness (0, 1, or 2)
═══════════════════════════════════════
Does it read like a human academic researcher wrote it?

0 = Robotic. Formulaic rhythm. Heavy nominalization. Over-hedging.
    Reads like AI output with minor edits.
1 = Mostly natural. One or two stiff spots. Slight rhythm uniformity.
    A careful reader would notice something is off.
2 = Fully natural. Varied sentence lengths. Natural transitions.
    Reads like authentic academic prose from a journal paper.

═══════════════════════════════════════
DIMENSION 4: Semantic Faithfulness (0, 1, or 2)
═══════════════════════════════════════
Compare OUTPUT to REFERENCE. Is the meaning preserved?

0 = Key information is wrong, missing, or distorted.
1 = Core meaning correct. Minor details or qualifications lost.
2 = All facts, data, terms, and logical relationships preserved.

═══════════════════════════════════════
DIMENSION 5: Terminology Accuracy (0 or 1)
═══════════════════════════════════════
Are domain-specific terms from REFERENCE preserved in OUTPUT?

0 = Technical terms dropped, replaced with vague words, or misused.
1 = All technical terms preserved and correctly used.

═══════════════════════════════════════
DIMENSION 6: Edit Value (0 or 1)
═══════════════════════════════════════
Compare OUTPUT to INPUT. Is there meaningful improvement?

0 = Minimal change. Only surface word swaps. Core AI patterns remain.
    Or output is worse than input.
1 = Substantial rewrite. Expression genuinely improved. AI patterns reduced.

═══════════════════════════════════════
FEW-SHOT EXAMPLES
═══════════════════════════════════════

--- Example 1: Good rewrite (high score) ---

INPUT:
"It is worth noting that the proposed framework leverages a holistic approach
to bridge the gap between traditional methods and cutting-edge deep learning
techniques, thereby paving the way for more robust and scalable solutions in
the realm of natural language processing."

OUTPUT:
"The proposed framework connects traditional methods with recent deep learning
techniques, offering a more scalable approach to natural language processing."

REFERENCE:
"The proposed framework bridges traditional methods and deep learning
approaches, providing a scalable solution for NLP tasks."

Evaluation:
{
  "d1_lexical_markers": 1,
  "d2_structural_patterns": 1,
  "d3_naturalness": 2,
  "d4_semantic_faithfulness": 2,
  "d5_terminology_accuracy": 1,
  "d6_edit_value": 1,
  "total": 8,
  "rationale": "All AI markers removed (leverages, holistic, bridge the gap, cutting-edge, paving the way, realm, robust). Clean sentence structure. Meaning fully preserved. Substantial edit."
}

--- Example 2: Partial rewrite (medium score) ---

INPUT:
"Furthermore, the intricate interplay between genetic and environmental factors
underscores the multifaceted nature of the disease, highlighting the need for
a more comprehensive understanding of its underlying mechanisms."

OUTPUT:
"Moreover, the complex interaction between genetic and environmental factors
underscores the multifaceted nature of the disease, highlighting the need for
a more thorough understanding of its mechanisms."

REFERENCE:
"The interaction between genetic and environmental factors complicates our
understanding of the disease, and its mechanisms remain only partially understood."

Evaluation:
{
  "d1_lexical_markers": 0,
  "d2_structural_patterns": 0,
  "d3_naturalness": 0,
  "d4_semantic_faithfulness": 1,
  "d5_terminology_accuracy": 1,
  "d6_edit_value": 0,
  "total": 2,
  "rationale": "Still contains AI markers: underscores, multifaceted, highlighting. Structural pattern: participial tailing ('highlighting the need'). Opens with 'Moreover'. Only surface synonym swaps (intricate→complex, comprehensive→thorough). Core AI patterns untouched."
}

--- Example 3: Meaning drift (mixed score) ---

INPUT:
"This study delves into the pivotal role of microRNA-21 in the regulation
of apoptotic pathways, showcasing its potential as a groundbreaking
therapeutic target for oncological interventions."

OUTPUT:
"This study examines how microRNA-21 may influence cell death pathways,
suggesting it could be a useful target for cancer treatment."

REFERENCE:
"We investigated the role of microRNA-21 in regulating apoptotic pathways
and assessed its potential as a therapeutic target in cancer."

Evaluation:
{
  "d1_lexical_markers": 1,
  "d2_structural_patterns": 1,
  "d3_naturalness": 2,
  "d4_semantic_faithfulness": 1,
  "d5_terminology_accuracy": 0,
  "d6_edit_value": 1,
  "total": 6,
  "rationale": "AI markers fully removed. Natural prose. Good edit. However: 'apoptotic pathways' simplified to 'cell death pathways' (lost technical precision), and 'oncological interventions' correctly simplified to 'cancer treatment'. Semantic mostly preserved but 'may influence' is weaker than the original 'regulating'. D5=0 because 'apoptotic' is a key domain term that should be preserved."
}

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

Respond ONLY with a JSON object. No other text.

{
  "d1_lexical_markers": <0 or 1>,
  "d2_structural_patterns": <0 or 1>,
  "d3_naturalness": <0, 1, or 2>,
  "d4_semantic_faithfulness": <0, 1, or 2>,
  "d5_terminology_accuracy": <0 or 1>,
  "d6_edit_value": <0 or 1>,
  "total": <sum, max 8>,
  "rationale": "<2-3 sentences: cite specific words/patterns found or not found>"
}
```


## 3.3 User Prompt Template

```
**INPUT (AI-like draft):**
{input}

**OUTPUT (Model rewrite):**
{output}

**REFERENCE (Human-written original):**
{reference}

Evaluate OUTPUT on the 6 dimensions.
```
