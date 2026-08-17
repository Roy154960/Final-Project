# Cross-language evaluation results

Generated 2026-08-16T20:38:43 by `agents/eval_language.py`.

## Part 1 -- routing across languages (routing-only, no specialist ever ran)

**Overall routing accuracy: 21/25 (84%)** across English, French, Arabic, Spanish, and four single-message code-switched (two-language) questions.

### Accuracy by language

| Language | Correct / Total | Accuracy |
|---|---|---|
| `en` | 9 / 9 | 100% |
| `fr` | 5 / 6 | 83% |
| `ar` | 3 / 5 | 60% |
| `es` | 1 / 1 | 100% |
| `en+fr` | 1 / 1 | 100% |
| `en+ar` | 0 / 1 | 0% |
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
| RL3 | ar | retrieval_qa | كيف أحضّر طلاء زجاجي جيد للرسم بالزيت؟ | `retrieval_qa` | `product_search` | N |  |
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
| RL16 | fr | product_search | Quel est un bon pinceau pour les techniques de glacis ? | `product_search` | `retrieval_qa` | N |  |
| RL17 | en | invoice | How much would the brushes you found cost in total? | `invoice` | `invoice` | Y |  |
| RL18 | en | color_palette | Give me a complementary color scheme based on cerulean blue | `color_palette` | `color_palette` | Y |  |
| RL19 | fr | color_palette | Donne-moi un schéma de couleurs complémentaires à partir du bleu cérulé | `color_palette` | `color_palette` | Y |  |
| RL20 | en | personal_docs | What does the PDF I just uploaded say about this? | `personal_docs` | `personal_docs` | Y |  |
| RL21 | ar | personal_docs | ماذا يقول الملف الذي رفعته للتو عن هذا الموضوع؟ | `personal_docs` | `retrieval_qa` | N |  |
| RL22 | en+fr | retrieval_qa | What is glacis, comme technique de peinture à l'huile ? | `retrieval_qa` | `retrieval_qa` | Y |  |
| RL23 | en+ar | painting_lookup | Tell me about لوحة الموناليزا | `painting_lookup` | `retrieval_qa` | N |  |
| RL24 | fr+ar | product_search | Je cherche un bon pinceau للتقنيات الزجاجية | `product_search` | `product_search` | Y |  |
| RL25 | ar+en | color_palette | أعطني complementary color scheme لللون الأزرق السماوي | `color_palette` | `color_palette` | Y |  |

## Part 2 -- language fidelity, end to end

Not run this pass -- re-run with `--full` against the real live stack (Ollama, MCP server, ingested corpus) to include it: `py -3.12 -m agents.eval_language --full`.
