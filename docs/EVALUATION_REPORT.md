# Bilingual Voice RAG Evaluation Report

## What this report covers

This is an honest review of the results saved in [`eval/results.json`](../eval/results.json), using the questions in [`eval/questions.jsonl`](../eval/questions.jsonl) and the scoring logic currently in [`eval/run_eval.py`](../eval/run_eval.py).

**This revision is not a new benchmark run.** The saved results name `minimax-m3.1-flash` as the generation model, but do not record a run timestamp, code revision, embedding endpoint, or complete configuration. The previous report dated the evaluation October 28, 2026; that date cannot be confirmed from the results file itself. These results should not be presented as measurements of the system after the recent fixes.

The short version: the saved run looks promising at finding the right source file, and several answers contain useful facts and citations. But it does not establish perfect evidence retrieval or reliable factual accuracy. There are incomplete answers, citation-format problems, and visible language-quality issues that the headline scores miss.

## What was tested

The question set contains 20 questions: 10 in English and 10 in Arabic. Sixteen are labelled answerable, and four deliberately ask about topics outside the supplied material.

The expected sources cover three files:

- `FEPS_student_handbook.pdf`: academic rules and grading information.
- `student_services_faq.txt`: fees, office locations, processing times, and administrative procedures.
- `diversity_school.csv`: enrollment figures.

Five questions are labelled as table questions. Two include a supplied conversation history to test a correction or follow-up. Three explicitly carry the `cross_language` tag. These are small, targeted checks—not broad coverage of bilingual conversation.

The evaluator retrieves up to five documents and sends the first three to a text-generation request. It does **not** measure a full microphone-to-speaker conversation. The earlier claim of 378 indexed chunks is not substantiated by the saved results, so it is not used here as a verified corpus count.

## What the saved scores actually mean

The counts below were checked against the individual saved question records. Timings were recalculated from those records; no new API calls were made for this report revision.

| Check | Saved result | How to read it |
|---|---:|---|
| Source-file Hit@1 | 16/16, 100% | Every answerable question has a recorded rank-one source-name match. This does not confirm the right page, section, row, or passage. |
| Source-file Hit@3 and Hit@5 | 16/16, 100% each | The same source-name check at larger cutoffs. |
| Source-file MRR | 1.000 | All 16 recorded first matches are at rank one. It has the same source-matching limitation. |
| Citation-marker detection | 15/16, 93.8% | A simple text check found a citation-like marker or source-related word. It does not verify citation accuracy. |
| “Groundedness” proxy | 13/16, 81.2% | For answerable questions, the script requires a detected citation and no detected refusal. It does not compare the answer with the reference facts. |
| Refusal-keyword detection | 3/4, 75.0% | Three unsupported questions matched the refusal keyword list. This is not a human-reviewed refusal-accuracy score. |
| Mean retrieval-stage time | 0.465 s | Recalculated across all 20 stored retrieval times. Median: 0.251 s; range: 0.223–2.561 s. |
| Mean generation-stage time | 3.739 s | Recalculated across all 20 stored generation times. Range: 2.109–6.096 s. Not spoken-response latency. |

I have removed the blanket “PASSED” labels. A score can meet a numerical target while still measuring something weaker than its name suggests.

### Why the retrieval score needs caution

The current `check_retrieval_hit` function matches source names using substring comparisons. It does not check the expected page, section, table, or CSV row. The results file also does not save the retrieved passages and their metadata for an independent inspection of each hit.

So the supported statement is **“all answerable questions recorded a rank-one source-file match,”** not “the correct evidence was always retrieved first.” Unsupported questions have empty expected-source lists, which the helper treats as a hit with rank zero. Those entries are excluded from the answerable-question retrieval scores and should not be interpreted as successful retrieval.

## A closer look at the answers

Here are the outcomes recorded for each question. “Proxy pass” below refers only to the script's citation-plus-no-refusal rule; it is not a manual factual endorsement.

| ID | Topic | What the saved output shows |
|---|---|---|
| Q01 | English grading scale | Gives GPA 3.6–4.0 but says the percentage range cannot be determined. Recorded proxy failure, not a complete answer. |
| Q02 | Transcript fee and processing time | Gives 100 EGP and 3–5 business days with `[S1]`. Proxy pass. |
| Q03 | Lost student ID | Gives 150 EGP and the Student Affairs Office, Room 101, with citations. Proxy pass. |
| Q04 | Credit-hour overload | Gives 21 hours and conditions, but uses `(S1)`, which the citation check misses. Recorded citation and proxy failure. |
| Q05 | Female enrollment at Phoenix-Arizona | Gives 134,722 with `[S1]`. Proxy pass. |
| Q06 | Total enrollment at Ivy Tech | Gives 91,179 with `[S1]`. Proxy pass. |
| Q07 | Arabic office designation | Gives إدارة شؤون الطلاب والسجلات and Room 104 with `[S2]`. Proxy pass. |
| Q08 | Corrected certificate request | Gives 50 EGP and two business days with `[S1]`. Proxy pass with the supplied history. |
| Q09 | Veterinary residency tuition | Says the context does not state the fee, but the refusal detector marks it false. A scoring limitation worth reviewing. |
| Q10 | JLPT graduation requirement | Says the evidence does not specify a minimum score. Refusal detected. |
| Q11 | Arabic lost-ID request | Gives the fee and procedure, but includes the stray Chinese text `丢了`. Proxy pass despite a clear language defect. |
| Q12 | Arabic enrollment certificate | Gives 50 EGP and two business days with a citation. Proxy pass. |
| Q13 | Arabic A+ grading question | Does not provide the requested A+ percentage or points. Includes `对应` and ends at “نسبة A+”. Recorded proxy failure. |
| Q14 | Arabic credit-hour limits | Distinguishes a regular maximum of 19 from an exceptional overload of 21. The earlier report's excerpt omitted that distinction. Proxy pass. |
| Q15 | Second-week refund | Gives 50% with `[S1]`. Proxy pass. |
| Q16 | Arabic office location and hours | Gives Room 104 and Sunday–Thursday, 9:00–2:30, with `[S1]`. Proxy pass. |
| Q17 | Arabic enrollment lookup | Gives 61,498, but also includes the awkward fragment `Bianca؟ لا.`. Proxy pass despite poor wording. |
| Q18 | Arabic urgent-service follow-up | Gives an additional 100 EGP and 24 business hours with citations. Proxy pass with the supplied history. |
| Q19 | Equestrian workshop fee | Says the fee is not in the available material. Refusal detected. |
| Q20 | Engineering admission requirements | Says the available evidence does not contain those requirements. Refusal detected. |

### What worked reasonably well

The saved fee and processing-time answers are concrete and generally easy to follow. For example, Q02 reports “100 EGP per copy” and “3–5 business days,” while Q08 switches to the enrollment certificate and reports “50 EGP per copy.” Q16 supplies an office location and opening hours in Arabic.

Both supplied-history correction questions produced answers that match the changed request. That is encouraging, but two scripted examples are not enough to claim reliable multi-turn voice conversation.

### What still needs attention

The grading questions are the clearest weak spot. Q01 answers only part of the request. Q13 does not provide the requested A+ values at all. The old report replaced that failure with an apparently successful answer; this revision does not repeat it.

Arabic fluency also needs a separate review. Q11, Q13, and Q17 contain visible foreign-language fragments or broken phrasing. These problems matter in a voice assistant even when a numerical fact is present.

Finally, the scoring can be misleading in both directions. Q04 loses credit because of citation punctuation. Q09 communicates that a fee is unavailable, but misses the refusal keyword list. Conversely, the proxy can approve an awkward or factually unchecked answer just because it includes `[S1]` and no refusal phrase.

## Recent debugging checks—separate from the benchmark

During the investigation in this workspace, we observed two additional generation problems:

1. The configured RAG credential returned HTTP 401, while the existing dedicated A2Agent credential completed a test request successfully.
2. After correcting the credential, a RAG request at the 150-token default returned empty text. Repeating the question with a 1,200-token budget produced a cited answer.

After the fixes, direct calls through the standalone `knowledge_base_rag` tool returned cited answers for the lost-ID fee question in English and the office-location/hours question in Arabic. The fee response reported **150 EGP**. The Arabic response still contained mixed-language wording and mentioned two differently named offices; a cited response alone does not settle answer quality.

The local `tests/` suite also passed **19 tests** with coverage collection disabled. These tests check application behavior, including error handling; they are not 19 independently evaluated RAG answers.

These are development checks observed in this chat, not a rerun of the 20-question benchmark. They were not written into `eval/results.json`, and no new benchmark percentages, reliability figures, or voice-latency measurements are claimed. Full microphone/TTS playback was not tested during those checks.

## Limits and the next useful evaluation

Before using this report as evidence of production readiness, I would address the following:

- **Make the benchmark reproducible.** Save the run timestamp, code revision, corpus identity, model settings, and retrieved passages. The current evaluator imports `build_rag_controller`, which is not defined in the current `src/voice/agent.py`, so its entry point needs repair before a fresh run.
- **Check evidence locations, not just filenames.** Validate the expected page, section, or row and inspect whether the returned passage supports the answer.
- **Review facts and language separately.** The script loads `reference_facts` but does not use them to score answers. A human-reviewed rubric should check completeness, numerical accuracy, source support, and Arabic fluency.
- **Record failure types explicitly.** Authentication errors, timeouts, empty generation, and genuinely absent evidence should not be counted as the same outcome.
- **Test the actual voice path.** Measure speech recognition, tool execution, first audible response, interruptions, and TTS quality in both languages. Text-generation timing does not answer those questions.

The saved data does not substantiate the earlier claims about Edge TTS cold-start timing, pre-warming improvements, or eliminated event-loop blocking. Those claims have been removed rather than presented as measured results.

## Bottom line

There is useful evidence here that the system can find relevant source files and produce cited answers for a small bilingual question set. There is also clear evidence of incomplete grading answers, imperfect scoring, and Arabic wording problems.

The fairest conclusion is that this is a promising development result—not proof of perfect retrieval, verified factual groundedness, or a fully evaluated voice experience. The recent fixes restored answers in two direct tool checks. A fresh, reproducible benchmark is still needed to measure the updated system.
