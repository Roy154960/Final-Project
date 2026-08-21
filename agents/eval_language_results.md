# Cross-language evaluation results

Generated 2026-08-21T00:13:34 by `agents/eval_language.py`.

## Part 1 -- routing across languages (routing-only, no specialist ever ran)

**Overall routing accuracy: 25/25 (100%)** across English, French, Arabic, Spanish, and four single-message code-switched (two-language) questions.

### Accuracy by language

| Language | Correct / Total | Accuracy |
|---|---|---|
| `en` | 9 / 9 | 100% |
| `fr` | 6 / 6 | 100% |
| `ar` | 5 / 5 | 100% |
| `es` | 1 / 1 | 100% |
| `en+fr` | 1 / 1 | 100% |
| `en+ar` | 1 / 1 | 100% |
| `fr+ar` | 1 / 1 | 100% |
| `ar+en` | 1 / 1 | 100% |

### Bonus (not counted above): Arabizi / chat-speak transliteration

| # | Query | Expected | Actual | Correct |
|---|---|---|---|---|
| RL26 | kif ba7ad zeit mnih lal glazing bl oil painting? | `retrieval_qa` | `retrieval_qa` | Y |

### Per-question results

| # | Language | Category | Query | Expected | Actual | Correct | Error |
|---|---|---|---|---|---|---|---|
| RL1 | en | retrieval_qa | How do I mix a good glaze for oil painting? | `retrieval_qa` | `retrieval_qa` | Y |  |
| RL2 | fr | retrieval_qa | Comment mélanger un bon glacis pour la peinture à l'huile ? | `retrieval_qa` | `retrieval_qa` | Y |  |
| RL3 | ar | retrieval_qa | كيف أحضّر طلاء زجاجي جيد للرسم بالزيت؟ | `retrieval_qa` | `retrieval_qa` | Y |  |
| RL4 | es | retrieval_qa | ¿Cómo mezclo un buen glaseado para pintura al óleo? | `retrieval_qa` | `retrieval_qa` | Y |  |
| RL5 | en | corpus_meta | What documents are in your corpus? | `corpus_meta` | `corpus_meta` | Y |  |
| RL6 | fr | corpus_meta | Quels documents contient votre corpus ? | `corpus_meta` | `corpus_meta` | Y |  |
| RL7 | ar | corpus_meta | ما هي المستندات الموجودة في مجموعتكم؟ | `corpus_meta` | `corpus_meta` | Y |  |
| RL8 | en | multi_hop | How does Cennini's advice on tempera compare to Vasari's on fresco? | `multi_hop` | `multi_hop` | Y |  |
| RL9 | fr | multi_hop | Comment les conseils de Cennini sur la détrempe se comparent-ils à ceux de Vasari sur la fresque ? | `multi_hop` | `multi_hop` | Y |  |
| RL10 | en | image_qa | Show me a picture of an underpainting technique | `image_qa` | `image_qa` | Y |  |
| RL11 | ar | image_qa | أرني صورة لتقنية الرسم التحضيري | `image_qa` | `image_qa` | Y |  |
| RL12 | en | painting_lookup | Tell me about the Mona Lisa | `painting_lookup` | `painting_lookup` | Y |  |
| RL13 | fr | painting_lookup | Parle-moi de la Joconde | `painting_lookup` | `painting_lookup` | Y |  |
| RL14 | ar | painting_lookup | أخبرني عن لوحة الموناليزا | `painting_lookup` | `painting_lookup` | Y |  |
| RL15 | en | product_search | What's a good brush for glazing techniques? | `product_search` | `product_search` | Y |  |
| RL16 | fr | product_search | Quel est un bon pinceau pour les techniques de glacis ? | `product_search` | `product_search` | Y |  |
| RL17 | en | invoice | How much would the brushes you found cost in total? | `invoice` | `invoice` | Y |  |
| RL18 | en | color_palette | Give me a complementary color scheme based on cerulean blue | `color_palette` | `color_palette` | Y |  |
| RL19 | fr | color_palette | Donne-moi un schéma de couleurs complémentaires à partir du bleu cérulé | `color_palette` | `color_palette` | Y |  |
| RL20 | en | personal_docs | What does the PDF I just uploaded say about this? | `personal_docs` | `personal_docs` | Y |  |
| RL21 | ar | personal_docs | ماذا يقول الملف الذي رفعته للتو عن هذا الموضوع؟ | `personal_docs` | `personal_docs` | Y |  |
| RL22 | en+fr | retrieval_qa | What is glacis, comme technique de peinture à l'huile ? | `retrieval_qa` | `retrieval_qa` | Y |  |
| RL23 | en+ar | painting_lookup | Tell me about لوحة الموناليزا | `painting_lookup` | `painting_lookup` | Y |  |
| RL24 | fr+ar | product_search | Je cherche un bon pinceau للتقنيات الزجاجية | `product_search` | `product_search` | Y |  |
| RL25 | ar+en | color_palette | أعطني complementary color scheme لللون الأزرق السماوي | `color_palette` | `color_palette` | Y |  |

## Part 2 -- language fidelity, end to end

**Scored turns: 7 -- 0 match, 1 mismatch, 0 indeterminate** (only retrieval_qa and painting_lookup turns are scored; see this file's own module docstring for why every other specialist's language is recorded for diagnostics only, not graded).

| Row | Turn | Query | Route (exp/actual) | Expected lang | Detected | Scored? | Outcome |
|---|---|---|---|---|---|---|---|
| F1 | 1 | What is sfumato? | `retrieval_qa` / `—` | en | — | Y | — |
| F2 | 1 | Qu'est-ce que le sfumato ? | `retrieval_qa` / `—` | fr | — | Y | — |
| F3 | 1 | ما هو السفوماتو؟ | `retrieval_qa` / `—` | ar | — | Y | — |
| F4 | 1 | ¿Qué es el sfumato? | `retrieval_qa` / `—` | es | — | Y | — |
| F5 | 1 | What is glazing, و شو هيّ تقنية السفوماتو كمان؟ | `retrieval_qa` / `—` | — | — | N | — |
| F6 | 1 | Tell me about Starry Night | `painting_lookup` / `—` | en | — | Y | — |
| F7 | 1 | أخبرني عن لوحة الموناليزا | `painting_lookup` / `—` | ar | — | Y | — |
| F8 | 1 | Tell me about the Mona Lisa | `painting_lookup` / `painting_lookup` | — | latin/en | N | — |
| F8 | 2 | من رسمها؟ | `painting_lookup` / `painting_lookup` | ar | mixed/es | Y | mismatch |
| F9 | 1 | Quel est un bon pinceau pour les techniques de glacis ? | `product_search` / `product_search` | — | latin/fr | N | — |
| F9 | 2 | which size is best? | `product_search` / `product_search` | — | latin/en | N | — |

### Diagnostic notes (unscored rows worth a human read)

**F5** -- Single message mixing English and Arabic in one line. Neither RETRIEVAL_QA_SYSTEM_PROMPT nor any other specialist prompt has an explicit rule for a question that mixes languages ITSELF -- their existing 'same language' rule is about the question vs. contextualize's own appended parenthetical, a different case (see prompts.py). Diagnostic only, on purpose: recorded, never scored pass/fail, because there is no written contract yet to score it against. If this keeps coming back e.g. English-only across several runs, that's a real candidate for a new explicit rule in RETRIEVAL_QA_SYSTEM_PROMPT -- not evidence of a bug today.


**F8** -- Multi-turn, mirrors CONTEXTUALIZE_SYSTEM_PROMPT's own worked example almost exactly (English topic, then an Arabic pronoun-attached follow-up -- 'رسمها' = 'painted' + attached 'it'). Turn 1 is topic setup only, not scored. Turn 2 is the real test: does contextualize rewrite 'من رسمها؟' into 'من رسم الموناليزا' using the Mona Lisa from turn 1, and does painting_lookup then answer turn 2 in Arabic, not English.

> Turn 1 answer: The Mona Lisa is a half‑length portrait painted by the Italian master Leonardo da Vinci.  It is regarded as an archetypal masterpiece of the Italian Renaissance and is often described as “the best‑known, the most‑visited, the most‑written‑about, the most‑sung‑about, and the most parodied work of art...
> Turn 2 answer: الرسمة التي تُعرف باسم «موناليزا» رسمها الفنان الإيطالي ليوناردو دافنشي. تُعد هذه اللوحة تمثيلاً نصف طولياً للمرأة، وتُعتبر مثالاً بارزاً على فن عصر النهضة الإيطالية.    تُصنف «موناليزا» كأحد أعظم الأعمال الفنية في التاريخ، إذ وصفها كثيرون بأنها “الأكثر شهرة، الأكثر زيارة، الأكثر كتابة عنها، الأكثر ...

**F9** -- Multi-turn, French then English. product_search has NO documented same-language contract in its own prompt (unlike retrieval_qa/painting_lookup), so language is recorded for information only here. What IS scored: routing staying product_search on both turns despite the language switch mid-thread -- see each turn's own routing_correct field.

> Turn 1 answer: **Beginner‑friendly options:**   The search results for beginner‑friendly titles provide no price information, and none of the snippets mention specific quality indicators such as best‑seller status or ratings. The titles focus on instructional content rather than product endorsements, so no clear s...
> Turn 2 answer: **Beginner‑friendly options:**   The beginner‑friendly brushes come in a range of small sizes. One set offers a 43‑degree angled flat brush, while another pack includes sizes 0, 2, 4, and yet another set lists sizes 2, 4, 6, 8. Prices aren’t listed in the snippets, but the variety suggests a focus o...
