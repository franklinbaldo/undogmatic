# Authority-Reliance Index for Brazilian Legal AI Outputs

## Quantifying Textual Patterns Associated with Professional Inadequacy

## Abstract

We propose the **Authority-Reliance Index (ARI)**, a metric for quantifying citation-explanation balance in legal text, specifically targeting patterns that Brazilian legal professionals recognize as inadequate argumentation. These patterns—heavy citation without substantive analysis—represent writing that competent attorneys would find professionally embarrassing to produce. While LLMs generating such patterns do not experience shame (they lack consciousness), these textual markers signal output quality issues analogous to what evokes professional discomfort in human practice.

We apply ARI to Brazilian higher court precedents (STF/STJ), comparing authority-only citations against explanation-focused paraphrases using embedding-based similarity to curated exemplars. In paired tests with 12 thesis statements, authority-only variants score significantly higher (M=56.0) than explained variants (M=48.0), p=0.001, effect size r=0.917. We propose validation pathways and discuss using ARI as a reward signal for training legal AI systems that avoid producing professionally inadequate outputs.

**Status:** Exploratory research requiring human expert validation before production deployment.

**Author positionality:** OAB-registered Brazilian attorneys operationalizing patterns we observe in professional practice.

---

## 1. Theoretical Framework: Professional Inadequacy Signals

### 1.1 The Shame Metaphor in Legal Practice

In Brazilian legal culture, producing arguments that cite precedents without explaining their application is professionally embarrassing. Experienced attorneys recognize such writing as marking the author as:
- Inexperienced or poorly trained
- Intellectually lazy
- Disrespectful to the court's time
- Attempting to conceal weak reasoning behind authority

This professional shame is a real phenomenon among human lawyers. When reviewing their own past work or being critiqued by colleagues, attorneys experience emotional discomfort upon recognizing these patterns in their writing.

### 1.2 Extending the Metaphor to AI Outputs

**Critical distinction:** LLMs do not experience shame. They are statistical models without consciousness, professional identity, or emotional states. When GPT-4 outputs "STF Tema 69" without explanation, no embarrassment occurs within the system.

However, we can meaningfully ask: "Does this AI output contain textual patterns that would evoke professional shame if a human attorney had produced them?"

This framing allows us to:
- Leverage professional judgment about quality standards
- Use shame as a heuristic for identifying inadequate outputs
- Maintain conceptual clarity about what we're actually measuring

**What ARI measures:** Textual similarity to patterns humans find professionally inadequate
**What ARI does not measure:** Emotional states in AI systems (which don't exist)

### 1.3 Why This Framing Matters

The shame metaphor is useful because:
1. **It's visceral**: Brazilian lawyers immediately recognize what we mean
2. **It's culturally grounded**: Reflects actual professional norms, not academic abstractions
3. **It's action-guiding**: If output would shame a human author, it needs improvement

But metaphors can mislead. We must be precise:
- "Shame-inducing patterns" ✓ (patterns that would cause shame in humans)
- "Measuring shame signals" ✓ (detecting textual markers of inadequacy)
- "AI experiencing shame" ✗ (conceptual error)
- "Shame in LLM outputs" ✗ (ambiguous—could imply AI has emotions)

### 1.4 Validation Requirements

To use shame as a quality heuristic, we must validate that:
1. Brazilian legal professionals share consensus on what constitutes "shame-worthy" citation patterns
2. The textual features we measure correlate with expert judgments of inadequacy
3. Outputs scoring low on ARI actually represent better legal writing
4. The metric doesn't conflate legitimate professional shorthand with genuine inadequacy

Without this validation, we're simply projecting our own discomfort onto patterns that may be acceptable or even preferred in actual practice.

---

## 2. Research Question

**Primary question:** Can we develop a quantitative metric that reliably identifies textual patterns in legal AI outputs that Brazilian legal professionals would find inadequate—specifically, citation without substantive explanation?

**Secondary questions:**
- Do Brazilian lawyers share consensus on what citation patterns are professionally embarrassing?
- Do these patterns correlate with argument effectiveness (case outcomes, judicial acceptance)?
- Can we distinguish inadequate citation from appropriate professional shorthand?

---

## 3. Methodology

### 3.1 Metric Design

The Authority-Reliance Index produces scores [0-100] where higher values indicate greater textual similarity to citation-heavy exemplars.

**Two backend implementations:**

1. **TF-IDF baseline:** Computes cosine similarity between input text and prototype vectors representing citation-heavy vs. explanation-rich patterns
2. **LLM evaluator:** Prompts Claude/GPT-4 to score texts on authority-reliance using structured criteria

**Prototype exemplars:** We curated Portuguese phrases representing patterns that, based on our professional experience, Brazilian lawyers routinely criticize as inadequate. [Documented in `data/prototypes.txt`]

### 3.2 Validation Approach

**Current status:** LLM-only validation (acknowledged methodological limitation)

**Required validation sequence:**
1. **Robustness testing** (pre-publication): Test for length confounds, citation-marker sensitivity, hybrid conditions
2. **Informal human validation** (2-3 weeks): 3-5 lawyer colleagues rate 15-20 samples
3. **Formal expert study** (3-6 months): 5-8 Brazilian lawyers, 50-100 samples, inter-rater reliability α>0.6, model-expert correlation r>0.7

### 3.3 Experimental Results (n=12 pairs)

| Metric | Authority-Only | Explanation-Focused |
|--------|----------------|---------------------|
| Mean ARI | 56.0 | 48.0 |
| Median ARI | 56.0 | 47.0 |
| Wilcoxon p | 0.001 | |
| Effect size | 0.917 (large) | |

**Interpretation:** The metric successfully distinguishes citation-heavy from explanation-rich presentation styles. Whether this distinction captures professional inadequacy requires human validation.

---

## 4. Applications (Conditional on Validation)

### 4.1 RLHF Reward Shaping

**If validated**, use ARI to penalize outputs containing shame-inducing patterns:

```python
reward = base_reward - λ × normalize(ARI)
```

**Hypothesis:** Training with ARI penalty will reduce generation of textual patterns that Brazilian legal professionals find inadequate, improving output quality for practitioner use.

**Critical requirement:** Human preference validation showing ARI-optimized outputs are actually preferred by lawyers.

### 4.2 Quality Monitoring

Track ARI distribution across:
- Model versions (detect regressions)
- Prompting strategies (A/B test approaches)
- User sessions (identify problematic outputs)

Display to users: "Explanation density: High/Medium/Low" based on ARI threshold.

### 4.3 Training Data Filtering

Score candidate training examples; prioritize low-ARI (explanatory) content for fine-tuning legal AI systems.

**Risk if unvalidated:** May filter legitimate professional writing while retaining verbose but empty explanations.

---

## 5. Limitations and Ethical Considerations

### 5.1 What This Metric Actually Measures (Open Question)

**We hypothesize ARI detects shame-inducing patterns.** Alternative interpretations:
- Text length (citations shorter than explanations)
- Lexical diversity (explanations use more varied vocabulary)
- Specific n-grams ("STF Tema", "Tema XXX")
- Academic vs. practitioner register

Robustness tests and expert validation will determine which interpretation is correct.

### 5.2 The Anthropomorphism Concern

Describing this work as "measuring shame in AI" invites misinterpretation. We are not claiming:
- LLMs experience emotions
- Embeddings encode affective states
- Statistical models have professional identities

We are claiming:
- Certain textual patterns evoke professional discomfort in human lawyers
- These patterns can be quantified
- AI systems can be trained to avoid producing them

This distinction must remain clear in all communications about this research.

### 5.3 Cultural Specificity

Professional shame is culturally constructed. What Brazilian lawyers find embarrassing may differ from:
- Other civil law jurisdictions
- Common law traditions
- Academic vs. practice contexts
- Appellate vs. transactional work

The metric operationalizes norms specific to Brazilian legal practice and may not transfer.

### 5.4 Gaming and Failure Modes

Models optimized for low ARI could learn to:
- Generate verbose but empty explanation-shaped text
- Add hedging phrases without substantive analysis
- Avoid legitimate citations to game the metric

**Mitigation:** Train discriminators to detect empty elaboration; use ARI as one signal among multiple quality metrics.

### 5.5 Deployment Without Validation Harms

Using this metric for RLHF before human validation could:
- Degrade output quality if the metric measures something different than intended
- Suppress appropriate professional shorthand
- Amplify LLM training biases rather than correcting them

**Commitment:** We will not deploy for production training until expert validation demonstrates r>0.7 correlation with professional judgment.

---

## 6. Validation Protocol

### Phase 1: Robustness Testing (Required Pre-Publication)

Test whether metric detects intended construct or spurious correlations:

1. **Hybrid condition:** Does "STF Tema 69: [explanation]" score high or low?
2. **Length confound:** When citations padded to match explanation length, do scores converge?
3. **Semantic substitution:** Does replacing "STF Tema" with synonyms change scores significantly?
4. **Multi-LLM consistency:** Do Claude, GPT-4, Gemini produce correlated scores (r>0.6)?

**Decision criterion:** If tests reveal major confounds, revise metric before human validation.

### Phase 2: Informal Human Validation

1. Show 15-20 text pairs to 3-5 Brazilian lawyer colleagues
2. Ask: "Which relies more on citation without explaining reasoning?"
3. Calculate correlation between human rankings and ARI scores
4. Interview about disagreements

**Decision criterion:**
- r>0.6: Proceed to formal study
- 0.4<r<0.6: Investigate and revise
- r<0.4: Metric likely measures something different; major revision required

### Phase 3: Formal Expert Study

1. Recruit 5-8 Brazilian lawyers (diverse practice areas)
2. Generate 50-100 sample texts (authority-only, explained, hybrid)
3. Independent ratings: "To what extent does this text represent the kind of writing that would be professionally embarrassing for a competent attorney to produce?"
4. Compute inter-rater reliability (target: α>0.6)
5. Measure model-expert correlation (target: r>0.7)

**Addressing low agreement:** If α<0.4, "professional inadequacy" may not be a unified construct—investigate whether disagreement clusters by practice area or document type.

### Phase 4: Outcome Validation

Connect to real-world consequences:
- Score actual briefs, track judicial acceptance rates
- Measure whether high-ARI arguments correlate with adverse case outcomes
- Intervention study: show lawyers their scores, measure quality improvements

Only positive outcome validation justifies claims that ARI measures something consequential for legal effectiveness.

---

## 7. Relationship to Existing Literature

### 7.1 Emotion in AI Research

Recent work explores how LLMs respond to emotional prompts and recognize emotions in text:

- **EmotionPrompt**: Adding emotional language to prompts improves task performance
- **Emotional intelligence tests**: LLMs can solve tests about recognizing emotions in scenarios
- **Sentiment analysis**: LLMs classify emotions expressed in human-written text

**Critical distinction:** These studies involve emotion recognition and response to emotional stimuli, not claims that AI experiences emotions. Our work continues this tradition—we're not measuring AI emotional states but rather textual patterns that evoke human emotional responses (professional shame).

### 7.2 Anthropomorphic Framing in AI

There's debate about using anthropomorphic language for AI systems:

**Arguments for:**
- Provides intuitive frameworks for understanding complex behavior
- Offers pragmatic terminology for discussing model outputs
- Can be useful metaphorically if clearly marked as such

**Arguments against:**
- Risks misleading users about AI capabilities
- Conflates pattern matching with genuine understanding
- May lead to inappropriate trust or emotional attachment

**Our position:** We use "shame" metaphorically to describe patterns humans find inadequate, while explicitly disclaiming that AI experiences shame. This follows the practice of using human-centric language ("the model understands," "the system knows") with clear caveats about what's literally true.

### 7.3 Legal AI Evaluation

Existing metrics focus on:
- Factual accuracy and hallucination detection
- Citation precision
- Reasoning transparency

ARI complements these by measuring explanatory depth—whether the system provides substantive analysis alongside citations. This addresses a quality dimension specific to legal practice that general-purpose metrics miss.

---

## 8. Philosophical Considerations

### 8.1 The Intentional Stance

Dennett's intentional stance suggests we can productively treat systems "as if" they have mental states when this yields predictive utility. Could we treat LLMs "as if" they experience professional shame?

**Our view:** Intentional descriptions are useful when they generate better predictions than mechanistic ones. For ARI, the mechanistic description ("the model produces tokens matching citation-heavy training patterns") fully explains the behavior. Adding intentional language ("the model feels shame about weak arguments") doesn't improve predictions—it just adds confusion.

However, saying "this output contains patterns that would shame a human author" is both accurate and useful for guiding quality judgments.

### 8.2 Metaphor vs. Literal Claim

Throughout this document, we use shame-adjacent language:
- "Professionally embarrassing patterns"
- "Shame-inducing outputs"
- "Inadequacy signals"

These are metaphorical extensions from human experience to AI-generated text. The literal claim is: "This text exhibits features that correlate with patterns Brazilian lawyers recognize as inadequate."

We believe the metaphorical framing is pedagogically useful for communicating with legal practitioners, but we maintain clear conceptual boundaries about what's literally true.

---

## 9. Conclusion

We propose ARI as a metric for identifying textual patterns in legal AI outputs that Brazilian legal professionals would find inadequate—specifically, citation without substantive explanation. These are patterns that would evoke professional shame if produced by a human attorney, though the AI generating them experiences no such emotion.

**What we've demonstrated:**
- The metric distinguishes citation-heavy from explanation-rich text (n=12, p=0.001)
- Implementation is reproducible and well-documented

**What requires validation:**
- Robustness tests for confounds (length, n-grams, hybrid conditions)
- Human expert agreement on what constitutes "shame-worthy" patterns
- Correlation between ARI scores and professional quality judgments (r>0.7 target)
- Outcome studies connecting ARI to argument effectiveness

**What we commit to:**
- No production deployment until expert validation succeeds
- Transparent reporting of all validation results, including negative findings
- Clear communication that we measure textual patterns, not AI emotional states

If validation succeeds, ARI could improve legal AI systems by reducing generation of outputs containing patterns that competent attorneys would find professionally embarrassing. If validation fails, we'll learn valuable lessons about the challenges of quantifying legal argumentation quality.

---

## 10. Transparency Statement

**Methodological limitations acknowledged:**
- Small sample (n=12)
- LLM-only validation to date
- Untested robustness for obvious confounds
- Unvalidated construct validity
- Cultural specificity to Brazilian legal practice

**What the metaphor means:** "Shame" refers to patterns that evoke professional discomfort in human lawyers, not emotional states in AI systems.

**Commitment:** We will revise all claims if validation reveals the metric measures something different than intended.

---

## Contact

**Repository:** [To be added]  
**License:** MIT  
**Status:** Exploratory research, pre-publication draft

---

**Authors:** [Names]  
**Affiliations:** OAB-registered Brazilian attorneys  
**Last updated:** [Date]  
**Version:** 1.0 (Exploratory)
