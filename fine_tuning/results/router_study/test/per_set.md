# Router comparison

Final test run: test sets included. Do not tune anything after this run.

## Accuracy

| Router | holdout_seen | holdout_new | team_ai | team_hand | human_written | Seconds per question | Cost per question (USD) |
|---|---|---|---|---|---|---|---|
| current method | 89.2% (222/249) | 76.9% (193/251) | 46.7% (28/60) | 34.8% (8/23) | 52.9% (37/70) | 0.0054 | 0.000000 |
| frozen + classifier | 94.0% (234/249) | 86.9% (218/251) | 60.0% (36/60) | 52.2% (12/23) | 60.0% (42/70) | 0.0055 | 0.000000 |
| fine-tuned | 98.8% (246/249) | 94.0% (236/251) | 60.0% (36/60) | 60.9% (14/23) | 62.9% (44/70) | 0.0036 | 0.000000 |
| frozen encoder | 76.7% (191/249) | 71.3% (179/251) | 45.0% (27/60) | 43.5% (10/23) | 55.7% (39/70) | 0.0035 | 0.000000 |
| fine-tuned, original data | 96.8% (241/249) | 98.0% (246/251) | 61.7% (37/60) | 43.5% (10/23) | 61.4% (43/70) | 0.0037 | 0.000000 |
| fine-tuned, 25% data | 98.4% (245/249) | 94.0% (236/251) | 61.7% (37/60) | 52.2% (12/23) | 67.1% (47/70) | 0.0043 | 0.000000 |
| fine-tuned, 50% data | 98.0% (244/249) | 94.8% (238/251) | 60.0% (36/60) | 56.5% (13/23) | 68.6% (48/70) | 0.0035 | 0.000000 |
| fine-tuned, distilbert-base-uncased | 96.0% (239/249) | 95.2% (239/251) | 60.0% (36/60) | 69.6% (16/23) | 65.7% (46/70) | 0.0037 | 0.000000 |
| fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | 97.6% (243/249) | 97.6% (245/251) | 58.3% (35/60) | 60.9% (14/23) | 68.6% (48/70) | 0.0066 | 0.000000 |
| llm small | 96.4% (240/249) | 98.8% (248/251) | 66.7% (40/60) | 87.0% (20/23) | 81.4% (57/70) | 0.7082 | 0.000077 |
| llm small + examples | 98.4% (245/249) | 99.2% (249/251) | 68.3% (41/60) | 82.6% (19/23) | 81.4% (57/70) | 0.6606 | 0.000110 |
| llm large | 98.8% (246/249) | 99.2% (249/251) | 73.3% (44/60) | 91.3% (21/23) | 88.6% (62/70) | 0.9505 | 0.001289 |
| llm large + examples | 99.6% (248/249) | 99.6% (250/251) | 70.0% (42/60) | 87.0% (20/23) | 87.1% (61/70) | 0.9409 | 0.003176 |
| live router | 97.2% (242/249) | 78.9% (198/251) | 56.7% (34/60) | 39.1% (9/23) | 54.3% (38/70) | 0.0061 | 0.000000 |

## Accuracy per route (holdout_new, team_ai, team_hand, human_written combined)

| Route | current method | frozen + classifier | fine-tuned | frozen encoder | fine-tuned, original data | fine-tuned, 25% data | fine-tuned, 50% data | fine-tuned, distilbert-base-uncased | fine-tuned, paraphrase-multilingual-MiniLM-L12-v2 | llm small | llm small + examples | llm large | llm large + examples | live router |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qualification_check | 78.5% (62/79) | 55.7% (44/79) | 81.0% (64/79) | 69.6% (55/79) | 79.7% (63/79) | 83.5% (66/79) | 82.3% (65/79) | 87.3% (69/79) | 77.2% (61/79) | 96.2% (76/79) | 92.4% (73/79) | 96.2% (76/79) | 94.9% (75/79) | 77.2% (61/79) |
| rules_info | 38.5% (20/52) | 50.0% (26/52) | 48.1% (25/52) | 46.2% (24/52) | 59.6% (31/52) | 48.1% (25/52) | 51.9% (27/52) | 42.3% (22/52) | 71.2% (37/52) | 67.3% (35/52) | 76.9% (40/52) | 78.8% (41/52) | 75.0% (39/52) | 48.1% (25/52) |
| sourcing_whatif | 57.1% (32/56) | 89.3% (50/56) | 89.3% (50/56) | 58.9% (33/56) | 94.6% (53/56) | 92.9% (52/56) | 92.9% (52/56) | 98.2% (55/56) | 96.4% (54/56) | 96.4% (54/56) | 100.0% (56/56) | 100.0% (56/56) | 100.0% (56/56) | 69.6% (39/56) |
| duty_cost | 89.5% (51/57) | 89.5% (51/57) | 91.2% (52/57) | 78.9% (45/57) | 91.2% (52/57) | 93.0% (53/57) | 91.2% (52/57) | 91.2% (52/57) | 94.7% (54/57) | 94.7% (54/57) | 93.0% (53/57) | 96.5% (55/57) | 96.5% (55/57) | 87.7% (50/57) |
| certification_draft | 80.0% (32/40) | 95.0% (38/40) | 87.5% (35/40) | 90.0% (36/40) | 95.0% (38/40) | 92.5% (37/40) | 87.5% (35/40) | 92.5% (37/40) | 92.5% (37/40) | 95.0% (38/40) | 90.0% (36/40) | 90.0% (36/40) | 92.5% (37/40) | 85.0% (34/40) |
| document_intake | 74.0% (37/50) | 86.0% (43/50) | 88.0% (44/50) | 60.0% (30/50) | 92.0% (46/50) | 92.0% (46/50) | 92.0% (46/50) | 90.0% (45/50) | 88.0% (44/50) | 86.0% (43/50) | 88.0% (44/50) | 92.0% (46/50) | 86.0% (43/50) | 80.0% (40/50) |
| escalate_human | 58.3% (21/36) | 75.0% (27/36) | 83.3% (30/36) | 33.3% (12/36) | 75.0% (27/36) | 77.8% (28/36) | 77.8% (28/36) | 75.0% (27/36) | 72.2% (26/36) | 86.1% (31/36) | 83.3% (30/36) | 88.9% (32/36) | 94.4% (34/36) | 52.8% (19/36) |
| out_of_scope | 32.4% (11/34) | 85.3% (29/34) | 88.2% (30/34) | 58.8% (20/34) | 76.5% (26/34) | 73.5% (25/34) | 88.2% (30/34) | 88.2% (30/34) | 85.3% (29/34) | 100.0% (34/34) | 100.0% (34/34) | 100.0% (34/34) | 100.0% (34/34) | 32.4% (11/34) |

## Most common mistakes

**current method:** out_of_scope -> escalate_human (18); sourcing_whatif -> escalate_human (16); rules_info -> escalate_human (14); qualification_check -> escalate_human (11); rules_info -> qualification_check (8)

**frozen + classifier:** qualification_check -> sourcing_whatif (19); rules_info -> qualification_check (12); rules_info -> out_of_scope (7); qualification_check -> out_of_scope (6); document_intake -> sourcing_whatif (5)

**fine-tuned:** rules_info -> qualification_check (8); qualification_check -> out_of_scope (8); rules_info -> out_of_scope (7); rules_info -> escalate_human (6); certification_draft -> escalate_human (5)

**frozen encoder:** sourcing_whatif -> qualification_check (11); qualification_check -> rules_info (10); rules_info -> qualification_check (10); document_intake -> sourcing_whatif (9); qualification_check -> duty_cost (9)

**fine-tuned, original data:** qualification_check -> sourcing_whatif (7); rules_info -> document_intake (7); rules_info -> qualification_check (6); rules_info -> out_of_scope (5); qualification_check -> out_of_scope (4)

**fine-tuned, 25% data:** rules_info -> escalate_human (10); rules_info -> qualification_check (5); qualification_check -> out_of_scope (4); rules_info -> out_of_scope (4); certification_draft -> escalate_human (3)

**fine-tuned, 50% data:** rules_info -> escalate_human (10); certification_draft -> escalate_human (5); qualification_check -> out_of_scope (5); rules_info -> qualification_check (4); rules_info -> duty_cost (4)

**fine-tuned, distilbert-base-uncased:** rules_info -> qualification_check (13); qualification_check -> out_of_scope (7); rules_info -> escalate_human (6); rules_info -> out_of_scope (5); escalate_human -> qualification_check (4)

**fine-tuned, paraphrase-multilingual-MiniLM-L12-v2:** qualification_check -> certification_draft (5); qualification_check -> sourcing_whatif (5); rules_info -> qualification_check (5); qualification_check -> out_of_scope (4); rules_info -> out_of_scope (4)

**llm small:** rules_info -> out_of_scope (6); rules_info -> qualification_check (6); document_intake -> out_of_scope (4); rules_info -> escalate_human (3); escalate_human -> qualification_check (2)

**llm small + examples:** rules_info -> out_of_scope (8); duty_cost -> out_of_scope (3); rules_info -> qualification_check (3); document_intake -> out_of_scope (3); certification_draft -> escalate_human (2)

**llm large:** rules_info -> qualification_check (6); rules_info -> escalate_human (3); certification_draft -> out_of_scope (2); escalate_human -> qualification_check (2); escalate_human -> document_intake (2)

**llm large + examples:** rules_info -> qualification_check (7); document_intake -> out_of_scope (3); document_intake -> escalate_human (2); qualification_check -> sourcing_whatif (2); rules_info -> escalate_human (2)

**live router:** out_of_scope -> escalate_human (18); sourcing_whatif -> escalate_human (12); rules_info -> qualification_check (10); qualification_check -> escalate_human (9); rules_info -> escalate_human (7)
