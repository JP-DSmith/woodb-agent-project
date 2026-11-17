# Router comparison: validation sets

Question sets: val_new_wording, val_new_style. 95% confidence intervals from 1,000 bootstrap resamples; p-values from an exact McNemar test on the same questions.

## 1. Does training change the model?

The current router doesn't train anything: it compares each question with stored examples using the pretrained model as it was downloaded. The fine-tuned router changes the model's weights. The frozen version trains only the final layer, so its encoder is identical to the pretrained one (weight change 0). Because that version reads only the first token's output, which MiniLM wasn't trained to use, the comparison also includes 'frozen + classifier': the unchanged pretrained model's averaged word vectors with a simple classifier on top. That is the fair test of what fine-tuning adds.

**The live router is the untrained download.** Of the 62 weight tables the live router's model file stores at full precision, **62 are identical** to a fresh download of the pretrained model (largest difference 0.0). Fine-tuning would have changed them.

| Version | Training questions | Weights trained | Best epoch | New wording macro-F1 | New style macro-F1 | Training time |
|---|---|---|---|---|---|---|
| fine-tuned | 6,264 | 22,716,296 of 22,716,296 | 2 | 98.2% | 94.0% | 124 s |
| frozen encoder | 6,264 | 3,080 of 22,716,296 | 20 | 65.0% | 63.4% | 189 s |
| fine-tuned, original data | 1,577 | 22,716,296 of 22,716,296 | 6 | 91.6% | 82.5% | 40 s |
| fine-tuned, 25% data | 1,566 | 22,716,296 of 22,716,296 | 5 | 93.7% | 91.6% | 39 s |
| fine-tuned, 50% data | 3,132 | 22,716,296 of 22,716,296 | 6 | 93.6% | 93.4% | 62 s |
| fine-tuned, distilbert-base-uncased | 6,264 | 66,959,624 of 66,959,624 | 3 | 90.6% | 88.4% | 133 s |
| fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | 6,264 | 117,656,840 of 117,656,840 | 4 | 95.3% | 93.7% | 230 s |

![Weight change](weight_change.png)

Route separation in the model's sentence space (silhouette, 600 validation questions): **0.14 before fine-tuning → 0.74 after** (about 0 means the routes are mixed together; towards 1 means clean, separate groups).

![Embeddings before and after](embeddings_before_after.png)

## 2. Choosing the training settings (validation only)

Every combination of learning rate and batch size was trained, keeping each run's best epoch. The best combination (top row) is used for all the fine-tuned versions below.

| Learning rate | Batch size | Best epoch | New wording macro-F1 | New style macro-F1 | Mean | Training time |
|---|---|---|---|---|---|---|
| 5e-05 | 32 | 2 | 98.2% | 94.0% | **96.1%** | 129 s |
| 2e-05 | 16 | 3 | 97.7% | 93.4% | **95.5%** | 227 s |
| 5e-05 | 16 | 2 | 97.2% | 93.6% | **95.4%** | 228 s |
| 0.0001 | 32 | 5 | 96.8% | 92.3% | **94.5%** | 126 s |
| 2e-05 | 32 | 6 | 95.3% | 92.0% | **93.6%** | 81 s |
| 0.0001 | 16 | 2 | 95.0% | 91.8% | **93.4%** | 232 s |

![Hyperparameter search](hyperparameter_search.png)

## 3. How it changed during training (fine-tuned, chosen settings)

| Epoch | Training loss | New wording acc. | New wording macro-F1 | New style acc. | New style macro-F1 | Encoder weight change |
|---|---|---|---|---|---|---|
| 0 | – | 13.7% | 8.8% | 13.5% | 8.8% | 0.0000 |
| 1 | 1.3491 | 87.9% | 89.0% | 86.4% | 87.3% | 0.0214 |
| 2 | 0.1272 | 97.8% | 98.2% | 92.9% | 94.0% | 0.0265 |
| 3 | 0.0293 | 96.2% | 96.4% | 93.4% | 93.9% | 0.0276 |
| 4 | 0.013 | 95.0% | 95.0% | 92.8% | 93.3% | 0.0279 |
| 5 | 0.0091 | 96.5% | 96.8% | 93.1% | 93.7% | 0.0281 |
| 6 | 0.0075 | 95.6% | 95.8% | 93.2% | 93.6% | 0.0281 |

![Learning curves](learning_curves.png)

## 4. Comparison

| Router | Questions | Accuracy (95% CI) | Macro-F1 (95% CI) | Seconds per question | AI cost per question |
|---|---|---|---|---|---|
| current method | 1623 | 75.8% (73.8%–77.8%) | 72.2% (70.0%–74.2%) | 0.0534 | $0.000000 |
| frozen + classifier | 1623 | 91.4% (90.0%–92.7%) | 91.6% (90.2%–93.0%) | 0.0473 | $0.000000 |
| fine-tuned | 1623 | 94.6% (93.5%–95.7%) | 95.4% (94.4%–96.4%) | 0.0266 | $0.000000 |
| frozen encoder | 1623 | 66.4% (64.2%–68.7%) | 64.0% (61.4%–66.4%) | 0.0258 | $0.000000 |
| fine-tuned, original data | 1623 | 85.3% (83.5%–87.1%) | 85.6% (83.6%–87.4%) | 0.0197 | $0.000000 |
| fine-tuned, 25% data | 1623 | 91.9% (90.6%–93.2%) | 92.3% (91.0%–93.6%) | 0.0265 | $0.000000 |
| fine-tuned, 50% data | 1623 | 93.1% (91.8%–94.3%) | 93.5% (92.2%–94.7%) | 0.0210 | $0.000000 |
| fine-tuned, distilbert-base-uncased | 1623 | 89.4% (87.9%–90.9%) | 89.2% (87.5%–90.8%) | 0.0227 | $0.000000 |
| fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | 1623 | 94.9% (93.7%–95.9%) | 94.3% (92.9%–95.5%) | 0.0370 | $0.000000 |
| llm small | 1623 | 97.1% (96.2%–97.8%) | 96.3% (95.2%–97.3%) | 0.6336 | $0.000078 |
| llm small + examples | 1623 | 94.2% (93.0%–95.4%) | 92.8% (91.4%–94.2%) | 0.6506 | $0.000113 |
| llm large | 1623 | 98.4% (97.8%–99.0%) | 97.9% (97.1%–98.7%) | 0.8536 | $0.001297 |
| llm large + examples | 1623 | 96.7% (95.9%–97.5%) | 95.9% (94.8%–96.9%) | 0.9023 | $0.003130 |

### Is the fine-tuned router really different?

| fine-tuned vs | Questions | Only fine-tuned right | Only other right | p-value | Real difference? |
|---|---|---|---|---|---|
| current method | 1623 | 340 | 35 | 0.0000 | yes (p < 0.05) |
| frozen + classifier | 1623 | 87 | 35 | 0.0000 | yes (p < 0.05) |
| frozen encoder | 1623 | 472 | 15 | 0.0000 | yes (p < 0.05) |
| fine-tuned, original data | 1623 | 160 | 9 | 0.0000 | yes (p < 0.05) |
| fine-tuned, 25% data | 1623 | 67 | 23 | 0.0000 | yes (p < 0.05) |
| fine-tuned, 50% data | 1623 | 45 | 21 | 0.0043 | yes (p < 0.05) |
| fine-tuned, distilbert-base-uncased | 1623 | 107 | 23 | 0.0000 | yes (p < 0.05) |
| fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | 1623 | 66 | 71 | 0.7327 | not shown |
| llm small | 1623 | 40 | 81 | 0.0002 | yes (p < 0.05) |
| llm small + examples | 1623 | 84 | 78 | 0.6946 | not shown |
| llm large | 1623 | 22 | 84 | 0.0000 | yes (p < 0.05) |
| llm large + examples | 1623 | 48 | 83 | 0.0028 | yes (p < 0.05) |

### F1 per route

| Route | current method | frozen + classifier | fine-tuned | frozen encoder | fine-tuned, original data | fine-tuned, 25% data | fine-tuned, 50% data | fine-tuned, distilbert-base-uncased | fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | llm small | llm small + examples | llm large | llm large + examples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qualification_check | 77.9% | 92.7% | 91.2% | 68.9% | 84.4% | 88.4% | 89.9% | 85.7% | 94.2% | 96.9% | 98.6% | 99.0% | 98.7% |
| rules_info | 77.6% | 92.1% | 96.9% | 67.6% | 85.6% | 92.1% | 93.0% | 94.6% | 97.8% | 96.1% | 96.4% | 98.8% | 99.4% |
| sourcing_whatif | 75.4% | 88.5% | 93.1% | 54.5% | 81.7% | 88.5% | 92.7% | 94.1% | 97.2% | 99.6% | 98.5% | 99.8% | 98.7% |
| duty_cost | 89.9% | 85.3% | 90.9% | 78.3% | 77.1% | 90.6% | 88.0% | 81.8% | 89.2% | 98.9% | 92.8% | 99.2% | 94.5% |
| certification_draft | 86.2% | 97.2% | 98.3% | 68.1% | 96.8% | 99.8% | 100.0% | 92.6% | 99.0% | 99.0% | 96.5% | 99.0% | 99.5% |
| document_intake | 88.8% | 91.6% | 99.1% | 71.6% | 88.5% | 97.1% | 99.4% | 97.6% | 99.1% | 96.7% | 91.8% | 96.4% | 95.1% |
| escalate_human | 21.8% | 92.1% | 96.4% | 32.2% | 85.0% | 94.9% | 98.2% | 85.0% | 96.3% | 88.7% | 92.4% | 93.6% | 94.0% |
| out_of_scope | 60.0% | 93.5% | 97.1% | 70.5% | 85.4% | 87.1% | 86.6% | 82.0% | 81.4% | 94.6% | 75.5% | 97.7% | 87.1% |

![F1 per route](f1_per_route.png)

![Confusion matrices](confusion_matrices.png)

![Cost vs quality](cost_vs_quality.png)

## 5. How much training data it needs

![Training size](training_size.png)

## 6. Economics

**A. Running cost per month** (what it costs to run the router itself)

| Router | Accuracy | Cost per question | 1,000/month | 100,000/month | 1,000,000/month |
|---|---|---|---|---|---|
| current method | 75.8% | $0.000000 | $0.00 | $0.00 | $0.00 |
| frozen + classifier | 91.4% | $0.000001 | $135.00 | $135.14 | $136.37 |
| fine-tuned | 94.6% | $0.000001 | $135.00 | $135.08 | $135.77 |
| frozen encoder | 66.4% | $0.000001 | $135.00 | $135.08 | $135.75 |
| fine-tuned, original data | 85.3% | $0.000001 | $135.00 | $135.06 | $135.57 |
| fine-tuned, 25% data | 91.9% | $0.000001 | $135.00 | $135.08 | $135.77 |
| fine-tuned, 50% data | 93.1% | $0.000001 | $135.00 | $135.06 | $135.61 |
| fine-tuned, distilbert-base-uncased | 89.4% | $0.000001 | $135.00 | $135.07 | $135.66 |
| fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | 94.9% | $0.000001 | $135.00 | $135.11 | $136.08 |
| llm small | 97.1% | $0.000078 | $0.08 | $7.80 | $78.00 |
| llm small + examples | 94.2% | $0.000113 | $0.11 | $11.30 | $113.00 |
| llm large | 98.4% | $0.001297 | $1.30 | $129.73 | $1,297.31 |
| llm large + examples | 96.7% | $0.003130 | $3.13 | $312.95 | $3,129.52 |

The fine-tuned routers have a fixed cost of about $135 a month whatever the volume: hosting ($15), 4 retrains a year and the one-time build (12 hours at $60) spread over 12 months. The prompted LLMs have no fixed cost, only a fee per question.

**B. Misrouted questions per month** (from each router's error rate on the validation questions)

| Router | Error rate | 1,000/month | 100,000/month | 1,000,000/month | Cost at $5.00 each, 100,000/month |
|---|---|---|---|---|---|
| current method | 24.2% | 242 | 24,214 | 242,144 | $121,072 |
| frozen + classifier | 8.6% | 86 | 8,626 | 86,260 | $43,130 |
| fine-tuned | 5.4% | 54 | 5,422 | 54,221 | $27,110 |
| frozen encoder | 33.6% | 336 | 33,580 | 335,798 | $167,899 |
| fine-tuned, original data | 14.7% | 147 | 14,726 | 147,258 | $73,629 |
| fine-tuned, 25% data | 8.1% | 81 | 8,133 | 81,331 | $40,665 |
| fine-tuned, 50% data | 6.9% | 69 | 6,901 | 69,008 | $34,504 |
| fine-tuned, distilbert-base-uncased | 10.6% | 106 | 10,598 | 105,977 | $52,988 |
| fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | 5.1% | 51 | 5,114 | 51,140 | $25,570 |
| llm small | 2.9% | 29 | 2,896 | 28,959 | $14,479 |
| llm small + examples | 5.8% | 58 | 5,792 | 57,917 | $28,959 |
| llm large | 1.6% | 16 | 1,602 | 16,020 | $8,010 |
| llm large + examples | 3.3% | 33 | 3,266 | 32,656 | $16,328 |

A misroute sends the question to the wrong specialist, so someone has to redirect or re-ask it. Its cost ($5.00, about five minutes of staff time) is an assumption, not a measurement, and at high volume it outweighs the cost of running any router. These error rates are from the validation questions, which are deliberately hard (new wording and new writing styles); everyday questions may be easier.

**C. When fine-tuning pays off** (monthly questions above which the 'fine-tuned' router is cheaper)

| Against | Running cost only | Including misroutes at $0.50 | Including misroutes at $5.00 |
|---|---|---|---|
| llm small | about 1,748,074 | never | never |
| llm small + examples | about 1,202,914 | about 68,855 | about 7,260 |
| llm large | about 104,124 | never | never |
| llm large + examples | about 43,149 | never | never |

'Never' means the fine-tuned router is not cheaper per question at that setting, so no volume pays for its fixed costs: on running cost alone, because the LLM's fee is lower than the fine-tuned router's compute; with misroutes, because the LLM's higher accuracy saves more than its fee costs.
