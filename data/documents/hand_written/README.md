# Hand-written unseen-layout documents (T26): TEST SET ONLY

19 supplier documents written by hand in layouts and languages the intake agent was not built from: plain
emails, shipping letters, CUSMA certifications, purchase order acknowledgements, CSV exports, text messages, and
invoices with different columns, in English, German, Spanish, Chinese, Italian and French. Each one has an
answer key: supplier, parts with their true origins, the problems planted on purpose, and the expected action.

| File | What it is |
|---|---|
| `T26_Unseen_Layout_Documents.xlsx` | The sheet the documents were written in (the source) |
| `make_from_sheet.py` | Turns the sheet into the files below |
| `documents/*.txt` | One file per document, the text exactly as written |
| `ground_truth.json` | The answer key |

Score the intake agent on them: `python evaluation/hand_documents_eval.py` (needs `OPENROUTER_API_KEY`; add
`--rules-only` to see what the rules alone do).

**Never use these documents to build, tune or prompt the intake agent.** They show how intake copes with
documents it has never seen; tuning on them would make that meaningless. If the intake agent is changed after
seeing these results, log the change (prompt change log, T35) and say so in the report.

All names, companies, addresses, phone numbers and email addresses are made up. 19 documents is a small set: the
results show which kinds of document cause trouble, not a precise accuracy figure.
