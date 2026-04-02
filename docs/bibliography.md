# IM-TRACE Bibliography

*All sources referenced across TESSERACT analysis runs, design decisions, and implementation rationale. Organized for arXiv paper citation.*

---

## Primary Benchmark References

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B1 | ARISE Network. "State of Clinical AI Report 2026." Stanford-Harvard multidisciplinary review, 500+ studies. Jan 2026. | 95% of clinical AI benchmarks measure accuracy alone; calls for claim-level grounding of reasoning traces | R2, R4 motivation, README framing |
| B2 | OpenAI. "HealthBench: Health question answering benchmark." May 2025. cdn.openai.com/pdf/healthbench_paper.pdf | 5,000 conversations, 262 physicians, 48,562 criteria. ChatGPT Health's live production benchmark (230M weekly queries) | Competitive positioning, case corpus reference |
| B3 | Chen et al. "CSEDB: Clinical Safety-Effectiveness Dual-Track Benchmark." Nature Digital Medicine, Dec 2025. | 32 experts, 23 specialties, 2,069 Q&A items, 17 safety + 13 effectiveness metrics | R3 design, evidence thresholds |
| B4 | [Authors]. "CLEVER: Clinical LLM Evaluation with Rubrics." JMIR AI, Dec 2025. ai.jmir.org/2025/1/e72153 | 3-rubric system (factuality, relevance, conciseness); randomized blind physician study | R1/R2 alignment, JMIR AI evidence bar |
| B5 | [Authors]. "MedR-Bench." Nature Communications, 2025. github.com/MAGIC-AI4Med/MedRBench | 1,453 structured cases; claims first quantitative reasoning process quality analysis | Competitive positioning — blocks broad novelty claim |
| B6 | [Authors]. "Q4Dx: Interactive Diagnostic Questioning Benchmark." 2025. | Interactive questioning efficiency under partial information | Adjacent comparator — different evaluation mode |

## Regulatory and Governance

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B7 | FDA. "Clinical Decision Support Software: Final Guidance." Jan 6/28, 2026. fda.gov | 4-criteria test for Non-Device CDS; Criterion 4 requires HCP independent review of reasoning basis | R4 regulatory alignment, primary commercial anchor |
| B8 | Honigman LLP. "FDA CDS 2026 Analysis." Feb 2026. honigman.com/alert-3236 | Criterion 4 softening: "summary of general approach sufficient for intended HCP"; time-criticality moved C3→C4 | S1 CAIER checklist update |
| B9 | Arnold & Porter. "FDA Updates CDS Software Guidance." Jan 2026. arnoldporter.com | Enforcement discretion for single clinically appropriate recommendation | Regulatory alignment docs |
| B10 | Jones Day. "Relaxing 2026: FDA Updates General Wellness and CDS Guidance." Jan 2026. jonesday.com | Confirmed Criterion 4 softening details | Regulatory mapping |
| B11 | CHAI / Joint Commission. "RUAIH: Responsible Use of AI in Healthcare." Sep 17, 2025. | First major US clinical AI governance framework from accreditation body; 7 governance domains | R14, governance alignment |
| B12 | Hooper Lundy. "JC and CHAI Release Governance Frameworks." 2025. hooperlundy.com | RUAIH analysis: formalized governance for deployment and oversight | RUAIH mapping |
| B13 | EU AI Act. "Article 9: Risk Management System." Effective Aug 2, 2026. artificialintelligenceact.eu/article/9/ | Healthcare AI classified high-risk; requires risk identification + misuse evaluation | R3 → Article 9(a), R4 → Article 9(b) |
| B14 | NIST. "AI 800-3: Statistical Framework for AI Benchmark Evaluation." Feb 2026. nist.gov | Distinguishes benchmark accuracy from generalized accuracy; requires uncertainty quantification | R9, academic credibility, paper methodology |
| B15 | AAAHC. "Governing Bodies Required to Provide Oversight of AI Quality." Mar 2026. | Clinical AI safety evaluation now a governance-level requirement at accredited organizations | Market validation |

## AI Evaluation Methodology

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B16 | [Authors]. "MonitorBench." Mar 30, 2026. | CoT monitorability DECREASES as model capability INCREASES | R2 technical rationale — surface fluency unreliable |
| B17 | [Authors]. "Automated Clinical Evaluation Rubrics." arXiv, Jan 2026. | Best automated grader achieves 60% Clinical Intent Alignment vs physician expert | 40% gap justification for physician evaluation |
| B18 | [Authors]. "Rethinking Medical Benchmarks for LLMs." arXiv, Aug 2025. arxiv.org/html/2508.04325v1 | Existing benchmarks have significant gaps in evaluating clinical reasoning processes | R4 motivation |
| B19 | Microsoft. "Healthcare AI Model Evaluator." github/learn.microsoft.com | Open-source framework with arena UI, multi-reviewer, model-as-judge, custom evaluators | Portability export design |
| B20 | [Authors]. "LLM-as-Judge for Clinical Evaluation." Nature Digital Medicine, Nov 2025. | Validated strong agreement for clinical evaluation tasks | LLM-judge implementation rationale |
| B21 | Tredence. "AI Red Teaming Guide." Feb 2026. tredence.com | Healthcare among sectors with highest AI red teaming demand | Market validation |
| B22 | [Authors]. "International AI Safety Report 2026." Feb 2026. | Clinical AI identified as high-risk domain requiring ongoing evaluation | Market validation |

## Clinical Reasoning Theory

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B23 | Norman GR. "Research in clinical reasoning: past history and current trends." Medical Education, 2005. | Multiple valid clinical reasoning pathways (hypothetico-deductive, pattern recognition, illness scripts) | R5 constraint-based gold standards |
| B24 | Eva KW. "What every teacher needs to know about clinical reasoning." Medical Education, 2005. | Expert reasoning follows multiple legitimate pathways | R5 medical rationale |
| B25 | Croskerry P. "A Universal Model of Diagnostic Reasoning." Academic Medicine, 2009. | Dual-process theory of clinical cognition (Type 1 intuitive + Type 2 analytical) | R4 medical rationale |
| B26 | Gebauer S. "FAILURE Framework for Clinical AI." Oct 2025. sarahgebauermd.substack.com | Fatigue, Automation bias, Information gaps, Limits, Unintended consequences, Real-world complexity, Emergent failures | R4 red teaming, CAIER S2 |
| B27 | [Authors]. "Nature Red-Teaming Protocol for Clinical AI." Nature, Mar 30, 2026. nature.com/articles/s41598-026-45719-3 | Error stratification (immediate/delayed/trust harm), dual-pronged testing | CAIER S2 red teaming |

## Psychometrics

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B28 | Linacre JM. "Many-Facet Rasch Measurement." MESA Press, 1989. | MFRM for separating rater severity, item difficulty, examinee ability | R10 MFRM design |
| B29 | Bond TG, Fox CM. "Applying the Rasch Model." Routledge, 3rd ed., 2015. | ~30 observations per facet for stable estimates | R10 sparse-data warnings |

## Market and Career Intelligence

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B30 | MatchDay Health. "Non-Clinical Physician Career Guide." Feb 2026. | Non-clinical physician salary: $180K-$350K+ | Income ceiling |
| B31 | 6figr/RemoteOK. "AI Red Teamer Compensation." Mar 2026. | AI red teamer: $119K-$265K; clinical premium 20-40% | Rate benchmarking |
| B32 | Articsledge. "AI Advisory Services Market." 2025-2026. | $11-14B in 2025, projected $90-257B by 2035 | Market sizing |
| B33 | GrowthList. "NYC Healthcare Startups." 2026. growthlist.co | 900+ funded NYC healthcare startups, 48% digital health | Outreach surface |
| B34 | CMSS. "CMSS Joins CHAI." Oct 2025. cmss.org | 800,000+ physicians via 50+ specialty societies | CHAI distribution |
| B35 | YC. "W26 Healthcare Companies." Mar 2026. ycombinator.com | 22 healthcare companies — largest vertical in YC W26 | Outreach targets |

## Ranking Theory (v1.2 References)

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B36 | [Authors]. "PARWiS: Spectral Ranking + Active Pair Selection." arXiv, 2026. arxiv.org/html/2603.01171v1 | Active pair selection under budget constraints for winner identification | R15 active scheduling |
| B37 | [Authors]. "SyncRank: Synchronization-Based Ranking." ACM, 2026. dl.acm.org/doi/10.1145/3783779.3783800 | Robust ranking from incomplete, noisy pairwise comparisons | R16 spectral baseline |
| B38 | Chen Y, Suh C. "Spectral MLE: Top-K Rank Aggregation." ICML, 2015. proceedings.mlr.press/v37/chena15 | Top-k identification from pairwise comparisons | R16 top-k confidence |
| B39 | [Authors]. "Fragility of LLM Leaderboards." arXiv, 2025. arxiv.org/html/2508.11847v3 | Rankings brittle under sparse preference perturbations | R17 frontier-set governance |
| B40 | [Authors]. "BLUR/AntiDote: Bi-Level Adversarial Training." 2026. | Bi-level formulation for systems where evaluated object adapts to evaluator | Future adversarial layer |
| B41 | Chatbot Arena / LMSYS. "Arena Rank: Open-Source Ranking." 2026. linkedin (lmarena) | Contextual Bradley-Terry; deployment-profile-aware ranking | R18 decision tiers |
| B53 | Artificial Analysis. "Intelligence Benchmarking Methodology." 2026. artificialanalysis.ai/methodology | Decision-tier leaderboard reporting (co-leader, top cluster, indistinguishable) | R18 leaderboard design |
| B54 | [Authors]. "Sequential Bayesian Experimental Design." PMC, 2014. pmc.ncbi.nlm.nih.gov/articles/PMC4181721 | MI-based criteria for cost-aware sequential experiment selection | R15 future acquisition function |
| B55 | NIST. "AI 800-2: Automated Benchmark Evaluation Best Practices." Jan 2026. nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-2.ipd.pdf | Validity, transparency, reproducibility pillars for AI evaluation | R8, R9 scientific rationale |
| B56 | [Authors]. "Contextual Bradley-Terry for LLM Evaluation." EACL, 2026. aclanthology.org/2026.eacl-long.291 | Stratified ranking by case features; profile-specific orderings | v1.2 stratified ranking |
| B57 | WIDA. "Validating Writing Scoring Scale Using Multi-Faceted Rasch." 2015. wida.wisc.edu | Standard MFRM validation framework for rubric development | R10, MFRM rater laboratory |
| B58 | [Authors]. "Correcting Human Labels for Rater Effects." arXiv, 2026. arxiv.org/html/2602.22585v1 | Importance of separating rater severity from output quality in annotation | R10, MFRM diagnostics |
| B59 | [Authors]. "Active Benchmark Design Under Budget." arXiv, 2026. arxiv.org/abs/2603.16756 | Models as hidden state, comparisons as observations, leaderboard as projection | v1.2 dynamical system framing |

## Job Market Evidence (TESSERACT Runs)

| ID | Citation | Key Finding | Used In |
|----|----------|-------------|---------|
| B42 | Outlier AI. "Medicine Expert." outlier.ai/experts/medicine, Mar 2026. | MD required, no license, up to $120/hr, remote | Income tier 1 |
| B43 | Remote AI Talent. "Physicians Remote AI Evaluation." Indeed, Apr 2026. | $150-250/hr, physician required, no license | Income tier 1 |
| B44 | Invisible Technologies. "Medicine Specialist AI Trainer." remotive.com, Jan 2026. | MD/DO required, $40-120/hr, no license | Income tier 1 |
| B45 | Hippocratic AI. "Life Sciences Evaluator Roles." jobs.ashbyhq.com, Apr 2026. | 4 evaluator positions, no license, $278M funded | Income tier 1 |
| B46 | Ambience Healthcare. "Physician AI Researcher." Multiple sources, 2026. | $205-300K, MD/DO only, H-1B confirmed, KLAS/CHIME Trailblazer | Breakthrough FTE target |
| B47 | Scale AI. "Medical Fellow." scale.com/careers, Feb 2026. | Board cert "required" but "open to residents" — logically inconsistent | Amber-lane target |
| B48 | VA/VHA. "Program Analyst (Informatics)." USAJobs, Mar 2026. | GS-0301/0343, no license required, $85-117K NYC | FTE federal pathway |
| B49 | Doctronic. "Senior Clinical Context Engineer." dynamitejobs, 2026. | $200-350K ceiling, requires completed residency | Ceiling signal |
| B50 | STAT News. "First Opinion Submission Guidelines." statnews.com | 800-1,200 words, financial disclosure, open submission | Publication pathway |
| B51 | NEJM Catalyst. "AI Implementation Special Issue." Mar 2026. | Active editorial interest in clinical AI methodology | Publication target |
| B52 | ATA. "RFI Response: Stakeholder Advisory Group." Feb 2026. | Calls for advisory group to define AI evaluation standards | Governance opportunity |
