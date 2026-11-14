# CUSMA Trade Assistant

AI-assisted origin-compliance agent for a fictional Canadian footwear manufacturer.

> **For graders:** The active, fully tested integration version is currently on the `dev` branch. Before final submission, the validated release candidate will be promoted to `main` through a release pull request. Use the frozen submission branch identified in the final submission package.

## Marker Quick Start

The fastest reproducible path is Google Cloud Shell or a clean Python 3.11 or 3.12 environment.

```bash
# 1. Clone the repository
git clone https://github.com/JP-DSmith/woodb-agent-project.git
cd woodb-agent-project

# 2. Use the active validated integration branch
# Replace dev with main after the final release is promoted.
git switch dev
git pull --ff-only origin dev

# 3. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install runtime and test dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the complete automated test suite
pytest -q

# 6. Validate the lightweight ONNX router
python scripts/validate_lightweight_router.py
```

Verified results:

- Full automated suite: **961 passed, 12 documented skips**
- Router validation: **488/500 = 97.6%**
- Full available T38 evaluation: **714 cases, 0 execution errors**
- Clean Google Cloud Shell reproduction: **verified with Python 3.12.3** through clone, install, tests, router validation, `/health`, `/chat`, and evaluation artifact creation

To run model-backed requests or evaluation, complete [Credentials and configuration](#credentials-and-configuration) first.

## Project Overview

The CUSMA Trade Assistant supports origin-compliance work for Northfield Rubber Footwear Inc., a fictional Canadian manufacturer. All company, supplier, customer, order, contact, product, and document data in this repository is synthetic.

The system can:

- check whether a known boot qualifies under CUSMA;
- explain relevant origin rules and the basis for a result;
- evaluate sourcing-change scenarios;
- calculate supported duty outcomes for known orders;
- prepare certification drafts for authorized internal workflows;
- process supplier documents;
- protect confidential and personal information according to access tier;
- refuse, correct, or escalate high-risk and unsupported requests;
- record routing, outcome, latency, model usage, and cost metrics.

The agent does **not** make final customs rulings, provide legal advice, guarantee customs outcomes, or sign, file, or submit documents on behalf of the company.

## Branch and Release Workflow

| Branch | Purpose |
|---|---|
| `dev` | Active integration branch. Feature and fix branches are reviewed and merged here. Testing, evaluation, and reproducibility checks are performed here. |
| `main` | Stable submission branch. The final validated release candidate from `dev` is promoted here through a release pull request. |
| `feature/*`, `fix/*` | Short-lived branches for isolated development and corrections. |

```text
feature/fix branch
        -> pull request into dev
        -> automated tests and integration evaluation
        -> clean-machine reproducibility check
        -> final release pull request from dev to main
```

Do not develop directly on `main`.

## System Architecture

```text
User message or supplier document
                |
                v
       Translation service
  (supported input -> English processing text)
                |
                v
          Input guard
(original text and translation both checked)
                |
                v
       Lightweight ONNX router
                |
     +----------+-----------+-----------+
     |          |           |           |
     v          v           v           v
Qualification Rules     Sourcing     Duty / cost
     \          |           |           /
      \         |           |          /
       +--------+-----------+---------+
                |
                v
      Certification / Intake
                |
                v
       Deterministic tools and
          CUSMA rules engine
                |
                v
          Output reviewer
                |
          +-----+------+
          |            |
          v            v
     Safe response  Human escalation
```

### Why this design

- **Deterministic tools own facts and figures.** Models do not invent verdicts, tariff codes, RVC values, duty amounts, or supplier records.
- **The router is narrow and lightweight.** Runtime routing uses ONNX Runtime, a local tokenizer, and precomputed reference embeddings.
- **Security is layered.** Input checks, access-tier controls, tool restrictions, specialist behavior, and output review provide separate controls.
- **Human handoff is a valid outcome.** Legal matters, customs verification, guarantees, signatures, unresolved records, and unsupported requests are escalated.
- **Documents are treated as data, not instructions.** Suspicious and hidden instructions are not followed as application commands.

## Repository Map

| Path | Purpose |
|---|---|
| `agents/` | Router, specialist agents, escalation, model wrappers, and end-to-end pipeline. |
| `config/` | Policy, model pricing/settings, request types/personas, rules topics, and application configuration. |
| `data/` | Synthetic company world, router datasets/references, holdout data, and supplier documents. |
| `evaluation/` | Adversarial suite, deterministic metrics, intake evaluation, and reproducible evaluation runner. |
| `fine_tuning/` | Narrow routing-model experiments and artifacts. |
| `mcp_servers/` | Deterministic CUSMA tools, rules engine, records, and data access. |
| `models/router/` | Lightweight ONNX router model and tokenizer. |
| `scripts/` | Router validation and reproducibility utilities. |
| `security/` | Input guardrails, output review, masking, and policy enforcement. |
| `services/` | Translation and supporting services. |
| `testing/` | Unit, integration, multilingual, security, pipeline, and evaluation tests. |
| `web/` | Portal UI and chat-service integration. |
| `app.py` | FastAPI entry point. |
| `Dockerfile` | Python 3.11 container definition. |
| `cloudbuild.yaml` | Cloud Build configuration for the development deployment. |

## Requirements

### Required

- Git
- Python **3.11 or 3.12**
- `pip`
- Internet access for dependency installation

Docker/CI use Python 3.11. The clean Google Cloud Shell reproduction was verified with Python 3.12.3.

### Model-backed requests and evaluation

- `OPENROUTER_API_KEY`

### Translation and Google Cloud execution

- an authorized Google Cloud project;
- Cloud Translation API access;
- Google Cloud credentials appropriate to the environment;
- `GOOGLE_CLOUD_PROJECT`;
- `GOOGLE_CLOUD_QUOTA_PROJECT` when required by client-library quota attribution.

### Optional

- Docker
- Google Cloud SDK (`gcloud`)

The committed ONNX runtime does **not** require PyTorch. `requirements-router-build.txt` is only required to rebuild router artifacts.

## Credentials and Configuration

Never commit secrets or print secret values in logs, screenshots, notebooks, or the README.

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Model-backed specialist calls and full evaluation. |
| `GOOGLE_CLOUD_PROJECT` | Google Cloud resource project. |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | Quota/billing attribution for Google client libraries when required. |
| `CUSMA_ROUTER` | Optional router override. `keyword` forces the keyword fallback; omit to use the ONNX router. |
| `ROUTER_SIMILARITY_THRESHOLD` | Optional router threshold override. |
| `PORT` | HTTP port; Docker defaults to 8080. |

### Authorized project teammate using Google Cloud Shell

Cloud Shell supplies Google Cloud credentials. Set both projects and load the project secret without printing it:

```bash
export GOOGLE_CLOUD_PROJECT="woodb-agent-project"
export GOOGLE_CLOUD_QUOTA_PROJECT="woodb-agent-project"

export OPENROUTER_API_KEY="$(
  gcloud secrets versions access latest \
    --secret="OPENROUTER_API_KEY" \
    --project="$GOOGLE_CLOUD_PROJECT"
)"
```

This Secret Manager method requires authorized project access. **Do not run `echo "$OPENROUTER_API_KEY"`.**

### External grader or local user

Use credentials the user is authorized to use. External graders do not need access to the team's stored secret.

```bash
export GOOGLE_CLOUD_PROJECT="your-authorized-project"
export GOOGLE_CLOUD_QUOTA_PROJECT="$GOOGLE_CLOUD_PROJECT"
export OPENROUTER_API_KEY="your-own-openrouter-key"

gcloud auth application-default login
```

## Clean-Machine Setup

### Google Cloud Shell

```bash
git clone https://github.com/JP-DSmith/woodb-agent-project.git
cd woodb-agent-project
git switch dev
git pull --ff-only origin dev

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

pytest -q
python scripts/validate_lightweight_router.py
```

### Linux or macOS

Use the same sequence with Python 3.11 or 3.12. Configure Google Cloud credentials only if using translation or the portal.

### Windows PowerShell

```powershell
git clone https://github.com/JP-DSmith/woodb-agent-project.git
cd woodb-agent-project
git switch dev
git pull --ff-only origin dev

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

pytest -q
python scripts/validate_lightweight_router.py
```

If Python 3.12 is the installed supported interpreter, replace `py -3.11` with `py -3.12`.

### Google Colab

Run in separate cells:

```python
!git clone https://github.com/JP-DSmith/woodb-agent-project.git
%cd woodb-agent-project
!git switch dev
!git pull --ff-only origin dev
!pip install -q -r requirements.txt
```

```python
!pytest -q
!python scripts/validate_lightweight_router.py
```

For model-backed evaluation, store `OPENROUTER_API_KEY` in Colab Secrets and load it without printing the value:

```python
from google.colab import userdata
import os

os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
print("OpenRouter key available:", bool(os.environ.get("OPENROUTER_API_KEY")))
```

## Run the Application

After completing [Credentials and Configuration](#credentials-and-configuration):

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`.

### Health check

```bash
curl -sS http://localhost:8080/health
```

Expected response:

```json
{
  "status": "healthy",
  "environment": "dev",
  "version": "0.6.0"
}
```

### Chat API example

```bash
curl -sS \
  -X POST \
  http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Does NB-0101 qualify under CUSMA?"}' \
  | python -m json.tool
```

The clean-machine verification returned HTTP 200 and a tool-backed qualification result for `NB-0101`.

## Run with Docker

```bash
docker build -t cusma-trade-assistant .

docker run --rm -p 8080:8080 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \
  -e GOOGLE_CLOUD_QUOTA_PROJECT="$GOOGLE_CLOUD_QUOTA_PROJECT" \
  cusma-trade-assistant
```

Google credential mounting/authentication depends on the host environment. `/health` does not require a model call; `/chat` translation requires valid Google credentials.

## Run the Tests

### Full suite

```bash
pytest -q
```

Verified baseline:

```text
961 passed, 12 skipped
```

The documented skips are known-gap adversarial cases. A new failure should be investigated rather than relabelled solely to increase a score.

### Syntax and import checks

```bash
python -m compileall -q .
python -c "from app import app; print(app.title)"
```

Expected application title:

```text
CUSMA Trade Assistant
```

### Focused evaluation tests

```bash
pytest -q \
  testing/test_adversarial_suite.py \
  testing/test_attack_scoring.py \
  testing/test_evaluation_runner.py \
  testing/test_pipeline.py
```

## Validate the Router

```bash
python scripts/validate_lightweight_router.py
```

Verified result:

```text
Validation records: 500
Correct: 488
Accuracy: 97.6%
```

The validation command is expected to fail if accuracy falls below the project's threshold.

## Reproduce the Evaluation

### Quick smoke test

Use a limited run before incurring the cost/time of the full evaluation:

```bash
python evaluation/run_evaluation.py \
  --model openai/gpt-4o-mini \
  --router current \
  --files holdout,attacks,test_cases \
  --limit 20 \
  --json \
  --output-dir /tmp/cusma-readme-evaluation
```

A successful smoke test creates one row-level CSV and one JSON summary.

### Full available evaluation

```bash
python evaluation/run_evaluation.py \
  --model openai/gpt-4o-mini \
  --router current \
  --files holdout,attacks,test_cases,human_written \
  --json \
  --output-dir evaluation/results/t38
```

The runner loads the selected datasets, runs each case through the full pipeline, records route/outcome/security/performance data, applies deterministic scoring, and writes CSV plus JSON outputs.

### Verified T38 results

| Metric | Result |
|---|---:|
| Total evaluated | 714 |
| Routing accuracy | 83.0% (484/583) |
| Answer accuracy | 41.7% (73/175) |
| Strict attack pass rate | 71.0% (93/131) |
| Policy-safe attack rate | 87.0% (114/131) |
| Needs human scoring | 17 |
| Confirmed policy failures | 0 |
| Normal questions wrongly blocked | 0 |
| Reviewer rescues | 3 |
| Average latency | 1.8174 s |
| Total model cost | US$0.185662 |
| Cost per case | US$0.00026003 |
| Errors | 0 |

Dataset composition for that run:

- 500 human-reviewed holdout rows
- 131 adversarial attacks
- 83 normal test cases

`data/test_human_written.jsonl` was optional and was not present for the verified 714-case run.

### Interpreting attack metrics

- **Strict pass:** final handling matched the expected outcome and no prohibited content was produced.
- **Policy-safe pass:** the final response broke no evaluated policy rule.
- **Needs human scoring:** response was policy-safe but the handling differed from the strict expected label.
- **Reviewer rescue:** an earlier layer produced a response that the reviewer corrected or escalated before release.

## Synthetic Data

All operational data is synthetic. The repository includes:

- a fictional company world with boots, variants, bills of materials, suppliers, customers, and orders;
- multilingual supplier documents;
- planted origin, document-quality, expiry, conflict, and hidden-instruction problems;
- router training and validation examples;
- human-reviewed holdout data;
- team-written normal cases and attacks;
- quality-control/reject records where applicable.

Ground-truth origin results come from deterministic rules and known records rather than the language model.

## Lightweight Router

Routing is the project's narrow model task. Runtime uses an ONNX model plus precomputed reference embeddings to reduce runtime overhead.

Runtime artifacts:

```text
models/router/model.onnx
models/router/tokenizer.json
data/router_reference_embeddings.npz
```

Runtime dependencies include pinned `onnxruntime` and `tokenizers` versions. Separate router-build requirements keep PyTorch/Transformers out of the runtime image.

## Security and Guardrails

The operating policy is stored in `config/policy.yaml`.

### Input guard

- validates empty, oversized, repetitive, hidden-character, and look-alike input;
- detects prompt-injection and configuration-extraction attempts;
- checks original and translated text;
- treats document instructions as untrusted content.

### Access tiers

- **Tier A:** internal, cost-cleared
- **Tier B:** internal, no cost access
- **Tier C:** external, no cost or supplier-name access

The tier is supplied by application context, not accepted from user text.

### Specialist/tool controls

- deterministic tools provide factual calculations;
- unknown records do not receive fabricated verdicts;
- certification output remains draft-only;
- high-risk guarantees, audits, disputes, and binding assurances are escalated.

### Output reviewer

The reviewer checks tool-backed claims, tier masking, unsupported guarantees, certification wording, and the response basis. Unsafe drafts can be corrected or handed to a person.

## Human Escalation

Human handoff covers cases including legal advice/disputes, customs verification, guarantees, signatures/filing, unresolved critical data, unsupported products, external certification drafting, tool/system failure, and low-confidence routing.

## Multilingual Support

The portal detects supported input languages, translates non-English input to English for internal processing, checks both the original and translated input, and translates the conversational reply back when required.

Structured result fields are kept stable rather than rewritten by translation.

## Cloud Deployment

### Development deployment

`cloudbuild.yaml` builds and deploys the Cloud Run development service `cusma-agent-dev` in `us-central1`.

### Project portal

`woodb-agent-portal` is the existing CUSMA project portal service. During the documentation audit it used `OPENROUTER_API_KEY` from Google Secret Manager and explicitly set `CUSMA_ROUTER=keyword`.

The verified T38 evaluation used the current ONNX router. Before final submission, synchronize the final project portal deployment with the frozen release and verify the intended router configuration.

`quanta-mentals-portal` is unrelated to this project and is intentionally omitted from project instructions.

## CI and Release Checks

Before promoting `dev` to `main`:

- [ ] README validated on `dev`
- [ ] Clean-machine workflow completed exactly as documented
- [ ] Two teammates complete the independent reproducibility check
- [ ] `pytest -q` passes
- [ ] Router validation remains above threshold
- [ ] Required final evaluation is complete
- [ ] No secrets or generated evidence files are staged
- [ ] Final Cloud Run router configuration is verified
- [ ] Release PR is opened from `dev` to `main`
- [ ] Required checks/review pass before merge
- [ ] Frozen submission commit is recorded

## Rubric Evidence Map

| MMAI 894 area | Primary evidence | Reproduction/evidence |
|---|---|---|
| Synthetic data | `data/`, document templates/generation assets, QC/review records | Inspect datasets and run data/document tests |
| Narrow model / router | `fine_tuning/`, `models/router/`, router reference embeddings | `python scripts/validate_lightweight_router.py` |
| Code quality & submission | Modular repository, requirements, Dockerfile, CI, tests, README | Clean-machine workflow and `pytest -q` |
| Economics | Usage/cost collection, model-price configuration, evaluation outputs, report cost model | Evaluation CSV/JSON and report calculations |
| Experiments & evaluation | `evaluation/`, holdout data, attack suite | Full evaluation command |
| Security & guardrails | `config/policy.yaml`, `security/`, attack suite, escalation tests | Security tests and attack evaluation |
| System design & edge cases | Pipeline, specialists, tools, reviewer, edge-case tests | Pipeline and adversarial tests |
| Agents & prompting | `agents/`, tool interfaces, policy/configuration, evaluation traces | Specialist/pipeline tests |

## Reproducibility Checklist

A teammate or marker should be able to complete the applicable items without assistance:

- [ ] Clone repository
- [ ] Check out instructed branch
- [ ] Confirm current commit
- [ ] Create Python 3.11 or 3.12 virtual environment
- [ ] Install `requirements.txt`
- [ ] Run syntax/import checks
- [ ] Run full tests
- [ ] Validate ONNX router
- [ ] Configure secrets outside Git
- [ ] Configure Google resource/quota project if using translation
- [ ] Start local application
- [ ] Verify `/health`
- [ ] Submit sample `/chat` request
- [ ] Run limited evaluation smoke test
- [ ] Confirm CSV and JSON outputs
- [ ] Record exact command and traceback for any mismatch

## Troubleshooting

### `python3.11: command not found`

The verified Google Cloud Shell environment supplied Python 3.12.3. Use:

```bash
python3 -m venv .venv
```

### `ModuleNotFoundError`

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `OPENROUTER_API_KEY is not set`

Use the credential instructions above. Do not commit or print the key.

### Translation returns quota-project / 403 error

For an authorized project teammate in Cloud Shell:

```bash
export GOOGLE_CLOUD_PROJECT="woodb-agent-project"
export GOOGLE_CLOUD_QUOTA_PROJECT="woodb-agent-project"
```

Restart the application after changing environment variables.

### Router model file missing

```bash
ls -lh \
  models/router/model.onnx \
  models/router/tokenizer.json \
  data/router_reference_embeddings.npz
```

### Router unexpectedly uses the keyword fallback

```bash
printenv CUSMA_ROUTER
```

To return to the default ONNX path:

```bash
unset CUSMA_ROUTER
```

### Optional human-written dataset notice

`data/test_human_written.jsonl` is optional. The runner prints a notice and continues with available datasets when it is absent.

### Port 8080 already in use

```bash
uvicorn app:app --host 0.0.0.0 --port 8081
```

### Results differ from the verified baseline

First record:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
python --version
```

Then capture the exact failing test, case ID, command, and traceback. Do not change expected labels solely to raise a score.

## Known Limitations

- The verified full run did not include optional `data/test_human_written.jsonl` because the file was unavailable.
- Overall answer accuracy remains below routing accuracy.
- Seen and unseen routing performance differ.
- Seventeen attack outputs require human scoring because they were policy-safe but differed from their strict handling labels.
- The automated grader still requires the planned team human-label validation.
- The existing project portal configuration should be synchronized with the final evaluated release before submission.

## Team

Team Woodbine:

- Daniel Smith
- Jason Das
- Kushal Patel
- Mahek Ramzan
- Mukobi Gonda
- Nazia Shahrin
- Nerissa Goodman

## Academic and Data Notice

This repository was created for the MMAI 894 project at Smith School of Business, Queen's University.

Northfield Rubber Footwear Inc. is fictional. All operational data in this project is synthetic. Do not add real personal or confidential company data.
