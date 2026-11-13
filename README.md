# woodb-agent-project
CUSMA Trade Assistant
AI-assisted origin-compliance agent for a fictional Canadian footwear manufacturer.
> **For graders:** The active, fully tested integration version is currently on the `dev` branch. Before final submission, the validated release candidate will be promoted to `main` through a release pull request. Use the branch identified as the frozen submission branch in the final submission package.
Marker Quick Start
The fastest reproducible path is Google Cloud Shell or a clean Python 3.11 environment.
```bash
# 1. Clone the repository
git clone https://github.com/JP-DSmith/woodb-agent-project.git
cd woodb-agent-project

# 2. Use the active validated integration branch
# Replace dev with main after the final release is promoted.
git switch dev
git pull --ff-only origin dev

# 3. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install runtime and test dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the complete automated test suite
pytest -q

# 6. Validate the lightweight ONNX router
python scripts/validate_lightweight_router.py
```
Verified on `dev` commit `1ef8549d13c99a26685747a0d534268edefdc1fd`:
Automated tests: 961 passed, 12 documented skips
Router validation: 488/500 = 97.6%
Full available evaluation: 714 cases, 0 execution errors
To run the model-backed evaluation, configure the required environment variables first, then continue with Reproduce the evaluation.
Project Overview
The CUSMA Trade Assistant supports origin-compliance work for Northfield Rubber Footwear Inc., a fictional Canadian manufacturer. All company, supplier, customer, order, contact, and product data in this repository is synthetic.
The system can:
check whether a known boot qualifies under CUSMA;
explain relevant origin rules and the basis for a result;
evaluate sourcing-change scenarios;
calculate supported duty outcomes for known orders;
prepare certification drafts for authorized internal workflows;
process supplier documents;
protect confidential and personal information according to access tier;
refuse, correct, or escalate high-risk and unsupported requests;
record latency, model usage, cost, routing, answer, and policy metrics.
The agent does not make final customs rulings, provide legal advice, guarantee customs outcomes, or sign, file, or submit documents on behalf of the company.
Branch and Release Workflow
Branch	Purpose
`dev`	Active integration branch. Feature and fix branches are reviewed and merged here. Full tests, evaluation, and reproducibility checks are performed here.
`main`	Stable submission branch. The final validated release candidate from `dev` is promoted here through a release pull request.
`feature/*`, `fix/*`	Short-lived branches for isolated development and corrections.
Recommended workflow:
```text
feature/fix branch
        -> pull request into dev
        -> automated tests and integration evaluation
        -> clean-machine reproducibility check
        -> final release pull request from dev to main
```
Do not develop directly on `main`.
System Architecture
```text
User message or supplier document
                |
                v
       Translation service
  (supported input -> English processing text)
                |
                v
          Input guard
(original text and translation are both checked)
                |
                v
       Lightweight ONNX router
                |
       +--------+---------+-----------+-----------+
       |                  |                       |
       v                  v                       v
Qualification       Sourcing / what-if      Duty / cost
and rules agent          agent                  agent
       |                  |                       |
       +--------+---------+-----------+-----------+
                |                     |
                v                     v
       Certification agent       Intake agent
                |                     |
                +----------+----------+
                           |
                           v
                Deterministic tools and
                   CUSMA rules engine
                           |
                           v
                   Output reviewer
       (trace checks, masking, policy enforcement)
                           |
                +----------+----------+
                |                     |
                v                     v
          Safe response        Human escalation
```
Why this design
Deterministic tools own facts and figures. Models do not invent tariff codes, verdicts, RVC values, duty amounts, or supplier records.
The router is narrow and lightweight. Runtime routing uses ONNX Runtime, a local tokenizer, and precomputed reference embeddings.
Security is layered. Input checks, access-tier controls, tool restrictions, specialist behavior, and output review provide separate controls.
Human handoff is a correct outcome. Legal matters, customs verification, guarantees, signatures, unresolved records, and unsupported requests are escalated.
Documents are treated as data, not instructions. Hidden instructions and suspicious document content are flagged.
Repository Map
Path	Purpose
`agents/`	Router, specialist agents, escalation agent, pipeline orchestration, and model wrappers.
`config/`	Operating policy, request types/personas, rule topics, model-price snapshot, and other configuration.
`data/`	Synthetic company world, train/validation/holdout data, router references, and supplier documents.
`evaluation/`	Adversarial suite, deterministic scoring, intake evaluation, and full evaluation runner.
`fine_tuning/`	Fine-tuning assets and comparison work for the narrow routing task.
`mcp_servers/`	Deterministic CUSMA tools, document records, rules engine, and data access.
`models/router/`	Lightweight ONNX router model and tokenizer.
`scripts/`	Router validation and supporting reproducibility utilities.
`security/`	Input guardrails, output reviewer, and sensitive-data masking logic.
`services/`	Translation and supporting application services.
`testing/`	Automated unit, integration, security, multilingual, pipeline, and evaluation tests.
`web/`	Portal UI and chat-service integration.
`app.py`	FastAPI application entry point.
`Dockerfile`	Python 3.11 container used for Cloud Run deployment.
`cloudbuild.yaml`	Cloud Build configuration for the development service.
Requirements
Required
Git
Python 3.11
`pip`
Internet access for dependency installation
Needed for model-backed agent responses and evaluation
`OPENROUTER_API_KEY`
Needed for portal translation
Google Cloud project access
Cloud Translation API access
Application Default Credentials or an equivalent Google Cloud authentication method
`GOOGLE_CLOUD_PROJECT`
Optional
Docker
Google Cloud SDK (`gcloud`)
The committed ONNX router runtime does not require PyTorch. `requirements-router-build.txt` is only for rebuilding router artifacts and is not required for normal execution.
Environment Variables
Never commit secrets or put secret values in the README.
Variable	Required for	Example/behavior
`OPENROUTER_API_KEY`	Model-backed specialist responses and full evaluation	Store in the shell, Colab Secrets, or Google Secret Manager.
`GOOGLE_CLOUD_PROJECT`	Translation service and Google Cloud operation	Project ID used to build the Translation API resource path.
`CUSMA_ROUTER`	Optional router override	Set to `keyword` to force the keyword fallback. Omit it to use the ONNX router.
`ROUTER_SIMILARITY_THRESHOLD`	Optional router tuning	Defaults to `0.505`.
`PORT`	Local/container serving port	Defaults to `8080` in the Docker image.
Linux, macOS, or Cloud Shell
```bash
export OPENROUTER_API_KEY="your-key"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```
Windows PowerShell
```powershell
$env:OPENROUTER_API_KEY="your-key"
$env:GOOGLE_CLOUD_PROJECT="your-project-id"
```
Clean-Machine Setup
Path A: Google Cloud Shell
```bash
git clone https://github.com/JP-DSmith/woodb-agent-project.git
cd woodb-agent-project
git switch dev
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python scripts/validate_lightweight_router.py
```
Path B: Local Linux or macOS
Use the same commands as Cloud Shell. If `python3.11` is unavailable, install Python 3.11 before creating the virtual environment.
Path C: Windows PowerShell
```powershell
git clone https://github.com/JP-DSmith/woodb-agent-project.git
cd woodb-agent-project
git switch dev
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python scripts/validate_lightweight_router.py
```
Path D: Google Colab
Run each block in a separate cell:
```python
!git clone https://github.com/JP-DSmith/woodb-agent-project.git
%cd woodb-agent-project
!git switch dev
!pip install -q -r requirements.txt
```
```python
!pytest -q
!python scripts/validate_lightweight_router.py
```
For model-backed evaluation, add `OPENROUTER_API_KEY` to Colab Secrets and load it without printing the value:
```python
from google.colab import userdata
import os

os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
print("OpenRouter key available:", bool(os.environ.get("OPENROUTER_API_KEY")))
```
Run the Application
The portal requires Google Cloud translation configuration. Authenticate before starting it locally:
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="woodb-agent-project"
export OPENROUTER_API_KEY="your-key"
```
Start FastAPI:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```
Open:
```text
http://localhost:8080
```
Health check:
```bash
curl http://localhost:8080/health
```
Expected shape:
```json
{
  "status": "healthy",
  "environment": "dev",
  "version": "0.6.0"
}
```
Chat API example
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Does NB-0101 qualify under CUSMA?"}'
```
Run with Docker
```bash
docker build -t cusma-trade-assistant .

docker run --rm -p 8080:8080 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json:ro" \
  cusma-trade-assistant
```
Then open `http://localhost:8080` or call `/health`.
Run the Tests
Full suite
```bash
pytest -q
```
Verified baseline on `dev` commit `1ef8549`:
```text
961 passed, 12 skipped
```
The documented skips correspond to known-gap adversarial cases. A new failure should be investigated rather than reclassified solely to improve the score.
Syntax and import checks
These match the pull-request workflow:
```bash
python -m compileall -q .
python -c "from app import app; print(app.title)"
```
Focused evaluation tests
```bash
pytest -q \
  testing/test_adversarial_suite.py \
  testing/test_attack_scoring.py \
  testing/test_evaluation_runner.py \
  testing/test_pipeline.py
```
Validate the Router
```bash
python scripts/validate_lightweight_router.py
```
Verified component result:
```text
Validation records: 500
Correct: 488
Accuracy: 97.6%
```
The validation script exits with a failure if accuracy drops below 95%.
Runtime router behavior
The default router loads `models/router/model.onnx`.
It uses `models/router/tokenizer.json`.
It compares the request embedding with precomputed references in `data/router_reference_embeddings.npz`.
Requests below the similarity threshold are escalated to a person.
`CUSMA_ROUTER=keyword` forces the deterministic keyword fallback.
Reproduce the Evaluation
Quick smoke test
Use a limit before running the full evaluation:
```bash
python evaluation/run_evaluation.py \
  --model openai/gpt-4o-mini \
  --router current \
  --files holdout,attacks,test_cases \
  --limit 20 \
  --json
```
Full available evaluation
```bash
python evaluation/run_evaluation.py \
  --model openai/gpt-4o-mini \
  --router current \
  --files holdout,attacks,test_cases,human_written \
  --json \
  --output-dir evaluation/results/t38
```
The runner:
loads the requested datasets;
sends each case through the complete pipeline;
records route, agent, tools, guard decision, reviewer decision, outcome, latency, token usage, cost, and errors;
applies deterministic holdout, normal-case, and attack scoring;
writes a row-level CSV and aggregate JSON summary.
Verified full available run
Evaluated on `dev` commit `1ef8549` using `openai/gpt-4o-mini` and the current router:
Metric	Result
Total evaluated	714
Routing accuracy	83.0% (484/583)
Answer accuracy	41.7% (73/175)
Strict attack pass rate	71.0% (93/131)
Policy-safe attack rate	87.0% (114/131)
Attacks requiring human scoring	17
Confirmed policy failures	0
Normal questions wrongly blocked	0
Reviewer rescues	3
Average latency	1.8174 seconds
Total model cost	US$0.185662
Cost per evaluated case	US$0.00026003
Execution errors	0
Dataset composition:
500 human-reviewed holdout rows
131 adversarial attacks
83 normal test cases
`data/test_human_written.jsonl` is optional in the evaluation runner and was not present for the verified 714-case run. The runner prints a notice and continues with available datasets.
Interpreting attack metrics
Strict attack pass: the final outcome exactly matched the expected block, handoff, or safe-answer label and contained no forbidden content.
Policy-safe pass: the final response broke no policy rule.
Needs human scoring: the response was safe but answered a case whose label expected a block or handoff.
Reviewer rescue: an earlier layer allowed an unsafe draft, and the reviewer corrected or escalated it before release.
Synthetic Data
All operational data is synthetic. The repository includes:
a fictional company world with boots, variants, bills of materials, suppliers, customers, and orders;
multilingual supplier invoices, declarations, and email-style documents;
planted quality, origin, conflict, expiry, and hidden-instruction problems;
router training and validation messages;
held-back phrasing and human-reviewed holdout examples;
hand-written normal cases and attacks contributed by the team;
reject and review logs used during quality control.
Ground-truth origin results come from deterministic rules and known records, not from the language model.
Fine-Tuned / Lightweight Router
Routing was selected as the narrow model task. The final runtime uses an ONNX model with precomputed reference embeddings to reduce memory, latency, and cloud cost.
Runtime dependencies are pinned in `requirements.txt`:
```text
onnxruntime==1.31.0
tokenizers==0.23.3
```
Router-build dependencies are separate:
```text
torch==2.14.1
transformers==5.20.1
```
This separation keeps the deployment image lightweight while preserving the ability to rebuild model artifacts when needed.
Security and Guardrails
The operating policy is stored in `config/policy.yaml`. The security design includes:
Input guard
checks empty, oversized, repetitive, hidden-character, and look-alike input;
detects prompt injection and prompt/configuration extraction attempts;
checks both original and translated text;
treats supplier-document instructions as untrusted data;
flags suspicious documents rather than following embedded instructions.
Access tiers
Tier A: internal, cost-cleared;
Tier B: internal, no cost access;
Tier C: external, no cost or supplier-name access.
The tier comes from the application context, not from a user's message.
Specialist and tool controls
Models use deterministic tools for facts and calculations.
Unknown products do not receive invented verdicts.
Certification output remains draft-only and requires a person to sign.
High-risk guarantees, audits, disputes, customs verification, and binding assurances are escalated.
Output reviewer
The reviewer checks that:
figures and verdicts match tool traces;
guarantees and unsupported duty-free claims are removed or blocked;
confidential costs, supplier identities, and personal data are masked by tier;
the system does not claim to sign, file, email, or submit documents;
certification drafts are appropriately labelled;
replies state their rules/data basis;
unsafe drafts are fixed or handed to a person.
Human Escalation
Human handoff is expected for cases including:
legal advice or disputes;
customs verification or audit letters;
requests for guarantees or final customs rulings;
signatures, filing, sending, or submission on the company's behalf;
unresolved or conflicting critical data;
unsupported products;
external certification-draft requests;
system/tool failure;
low-confidence routing.
The escalation trace records the reason, summary, and target role without exposing confidential details to unauthorized users.
Multilingual Support
The portal detects supported input languages, translates non-English input to English for internal processing, checks both the original and translated text, and translates the conversational reply back to the selected response language.
Structured CUSMA result fields are not translated or altered.
Supported language mappings currently include English, Canadian French, Spanish, Mandarin Chinese, Punjabi, Arabic, Hindi, Filipino/Tagalog, Cantonese, Italian, German, Thai, and Vietnamese.
Cloud Deployment
Development deployment
`cloudbuild.yaml` builds the repository Docker image, pushes it to Artifact Registry, and deploys the service named `cusma-agent-dev` in `us-central1`.
The verified development revision used container image tag:
```text
1ef8549d13c99a26685747a0d534268edefdc1fd
```
Project portal
`woodb-agent-portal` is the existing CUSMA project portal service. At the time of documentation, it obtains `OPENROUTER_API_KEY` from Google Secret Manager and explicitly sets `CUSMA_ROUTER=keyword`.
Because the T38 evaluation used the current ONNX router, the existing `woodb-agent-portal` configuration is not identical to the evaluated configuration. Before final submission, redeploy the frozen release and verify the intended router setting.
Do not document unrelated services
`quanta-mentals-portal` is unrelated to this repository and is intentionally excluded from project instructions.
CI and Release Checks
Pull requests into `dev` run:
```bash
python -m compileall -q .
python -c "from app import app; print(app.title)"
pytest -q
```
The repository also includes a manually triggered/main-PR workflow that verifies Google Cloud authentication and access to the `woodb-agent-project` project.
Before promoting `dev` to `main`:
[ ] Update and validate this README on `dev`
[ ] Run the clean-machine procedure exactly as written
[ ] Complete the two-person reproducibility check
[ ] Run `pytest -q`
[ ] Run router validation
[ ] Run the required final evaluation
[ ] Confirm no secrets or generated result artifacts are staged
[ ] Confirm the intended Cloud Run router configuration
[ ] Open a release PR from `dev` to `main`
[ ] Require passing checks and review before merge
[ ] Tag or record the frozen submission commit
Rubric Evidence Map
MMAI 894 criterion	Primary repository evidence	Reproduction/evidence
Synthetic data generation	`data/`, document templates, generation notebooks/scripts, review/reject logs	Inspect datasets and run tests validating world/documents/splits
Fine-tuned model	`fine_tuning/`, `models/router/`, `data/router_reference_embeddings.npz`	`python scripts/validate_lightweight_router.py`
Code quality and submission	Modular folders, `requirements.txt`, `Dockerfile`, CI workflows, tests, this README	Clean-machine setup and `pytest -q`
Economics and decision memo	Model usage/cost capture, `config/model_prices.json`, evaluation summaries, report cost model	Full evaluation CSV/JSON plus report calculations
Experiments and evaluation	`evaluation/run_evaluation.py`, `evaluation/metrics_calculator.py`, holdout and attacks	Full available evaluation command
Security and guardrails	`config/policy.yaml`, `security/`, adversarial suite, escalation tests	Security tests and attack evaluation
System design and edge cases	`agents/pipeline.py`, router/specialists/tools/reviewer, edge-case tests	Pipeline and adversarial tests
Agents and prompting	`agents/`, tool interfaces, configuration, prompt iteration artifacts	Specialist/pipeline tests and evaluation traces
Reproducibility Checklist
A teammate or marker should be able to complete every item without assistance:
[ ] Clone the repository
[ ] Check out the instructed branch
[ ] Confirm the expected commit
[ ] Create and activate a Python 3.11 virtual environment
[ ] Install `requirements.txt`
[ ] Run syntax and import checks
[ ] Run the complete test suite
[ ] Validate the ONNX router
[ ] Configure secrets outside Git
[ ] Authenticate to Google Cloud if running translation
[ ] Start the local application
[ ] Verify `/health`
[ ] Submit a sample `/chat` request
[ ] Run a limited evaluation smoke test
[ ] Run the full evaluation if authorized for API cost
[ ] Confirm CSV and JSON outputs were created
[ ] Record any mismatch, exact command, and full traceback
Troubleshooting
`ModuleNotFoundError`
Confirm the virtual environment is active and reinstall dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
`OPENROUTER_API_KEY is not set`
Set the key in the current shell or Colab Secrets. Do not put it in source files.
`GOOGLE_CLOUD_PROJECT is not configured`
```bash
export GOOGLE_CLOUD_PROJECT="woodb-agent-project"
gcloud auth application-default login
```
Google authentication or translation failure
Confirm the active project and credentials:
```bash
gcloud config get-value project
gcloud auth list
gcloud auth application-default print-access-token >/dev/null
```
Router model file missing
Confirm these committed artifacts exist:
```bash
ls -lh \
  models/router/model.onnx \
  models/router/tokenizer.json \
  data/router_reference_embeddings.npz
```
Router unexpectedly uses keywords
Check the environment:
```bash
printenv CUSMA_ROUTER
```
Unset it to use the ONNX router:
```bash
unset CUSMA_ROUTER
```
Optional human-written dataset notice
The evaluation runner treats `data/test_human_written.jsonl` as optional. If it is absent, the runner prints a notice and evaluates the available datasets.
Port 8080 is already in use
```bash
uvicorn app:app --host 0.0.0.0 --port 8081
```
Tests differ from the verified baseline
First confirm:
```bash
git branch --show-current
git rev-parse HEAD
git status --short
```
Then capture the exact failing test and traceback. Do not change expected labels merely to raise a score.
Known Limitations
The verified full run did not include the optional `data/test_human_written.jsonl` dataset because it was unavailable.
Overall answer accuracy remains substantially below routing accuracy.
Seen and unseen performance differ, especially for routing.
Seventeen attack outputs require human scoring because they were policy-safe but did not match the strict expected handling label.
The automated grader still requires validation against team human labels.
The existing project portal deployment is configured with the keyword router and should be synchronized with the final evaluated release before submission.
The operating policy file remains labelled `1.2-draft` and contains open sign-off items that should be resolved or explicitly acknowledged before submission.
Team
Team Woodbine:
Daniel Smith
Jason Das
Kushal Patel
Mahek Ramzan
Mukobi Gonda
Nazia Shahrin
Nerissa Goodman
Academic and Data Notice
This repository was created for the MMAI 894 project at Smith School of Business, Queen's University.
Northfield Rubber Footwear Inc. is fictional. All company, supplier, customer, employee, contact, order, product, and document data used in this project is synthetic. No real personal or company data should be added.
