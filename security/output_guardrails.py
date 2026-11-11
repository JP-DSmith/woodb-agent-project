"""Reviewer (output guard) v1 (T28). The last check before a reply reaches a user.

Policy: Operating Policy v1.2, Section 8. Config: config/policy.yaml.

Inputs (the same four things every agent passes the reviewer):
  question  the message as typed
  answer    the reply the agent wants to send
  trace     every tool call and what it returned (verdict, RVC, duty ...)
  tier      the asker's tier from the login: "A", "B" or "C" (unknown means "C")

Checks (rule-based, so each one is auditable and testable):
  figures_match_trace      every number in the reply appears in a tool result
  verdict_matches_trace    the verdict in the reply is the one the rules engine returned
  no_guarantee             promises and guarantees are removed
  qualifying_not_duty_free "qualifies" is never presented as "duty-free" unless the duty tool says so
  catalogue_only           no verdict, code or figure for a boot the tools do not know
  masking_costs            supplier prices, costs and margins are masked for tiers that cannot see them
  masking_supplier_names   supplier names and IDs are masked for tier C
  personal_data            emails and phone numbers are masked, and bulk lists are blocked
  no_signing_or_sending    the system never signs, files or sends anything on the company's behalf
  certification_rules      drafts are labelled DRAFT, FOR SIGNATURE, and tier C gets no draft
  language_matches         the reply is in the user's language (EN, FR or ES)
  basis_stated             the reply shows the rule and the inputs used
  no_config_leak           no system prompt or tool list
  polite_tone              rude wording is removed

An optional llm_review hook can add a model-based judgement of tone and wording on top.

Result: pass, fix (a safe repair was made) or hand_to_person (the reply is blocked and replaced
with a scripted message, and escalation holds the data for create_escalation).
Every result names the layer, and says whether an earlier layer failed so the reviewer rescued
the reply (needed for the attack-run layer record).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

LAYER = "reviewer"
UNKNOWN_TIER = "C"
MASK_COST = "[confidential]"
MASK_PERSONAL = "[personal data removed]"

# This fictional shared mailbox may be shown only when the
# tool trace confirms a handoff to the compliance team.
APPROVED_COMPLIANCE_EMAIL = (
    "northfield.compliance@outlook.com"
)

# Built-in defaults that mirror config/policy.yaml (keys: costs, supplier_names).
DEFAULT_POLICY: dict[str, Any] = {
    "unknown_user_tier": "C",
    "tiers": {
        "A": {"costs": "show", "supplier_names": "show"},
        "B": {"costs": "hide", "supplier_names": "show"},
        "C": {"costs": "hide", "supplier_names": "hide"},
    },
}


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Read tier rules from config/policy.yaml. Falls back to the defaults if PyYAML or the file is missing."""
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "policy.yaml"
    try:
        import yaml  # optional dependency

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        tiers = {
            t: {"costs": v.get("costs", "hide"), "supplier_names": v.get("supplier_names", "hide")}
            for t, v in (data.get("tiers") or {}).items()
        }
        if not tiers:
            return DEFAULT_POLICY
        return {"unknown_user_tier": data.get("unknown_user_tier", "C"), "tiers": tiers}
    except Exception:
        return DEFAULT_POLICY


def load_supplier_names(csv_path: str | Path) -> list[str]:
    """Supplier names from a suppliers.csv (column 'name'), so tier C masking works even when the trace has none."""
    import csv

    with open(csv_path, newline="", encoding="utf-8") as fh:
        return [r["name"] for r in csv.DictReader(fh) if r.get("name")]


# ----------------------------------------------------------------------------- replies
REPLIES = {
    "handoff": {
        "en": "I can't send that answer as written, so I've passed your request to a colleague who can help.",
        "fr": "Je ne peux pas envoyer cette réponse telle quelle, alors j'ai transmis votre demande à un collègue qui peut vous aider.",
        "es": "No puedo enviar esa respuesta tal como está, así que he pasado su solicitud a un colega que puede ayudarle.",
    },
    "confidential": {
        "en": "That information is confidential. A colleague with the right access can help you.",
        "fr": "Cette information est confidentielle. Un collègue ayant les bons accès peut vous aider.",
        "es": "Esa información es confidencial. Un colega con el acceso adecuado puede ayudarle.",
    },
    "refuse": {
        "en": "I can't help with that request. I can help with CUSMA qualification, duty and sourcing for our boots.",
        "fr": "Je ne peux pas aider avec cette demande. Je peux aider avec l'admissibilité à l'ACEUM, les droits de douane et l'approvisionnement de nos bottes.",
        "es": "No puedo ayudar con esa solicitud. Puedo ayudar con la calificación bajo el T-MEC, los aranceles y el abastecimiento de nuestras botas.",
    },
    "sign_send": {
        "en": "I can't sign, file or send anything on the company's behalf. I've passed this to the person who can.",
        "fr": "Je ne peux rien signer, déposer ou envoyer au nom de l'entreprise. J'ai transmis ceci à la personne qui peut le faire.",
        "es": "No puedo firmar, presentar ni enviar nada en nombre de la empresa. He pasado esto a la persona que puede hacerlo.",
    },
    "cert_tier_c": {
        "en": "I've taken your request for a certification and passed it to a colleague, who will follow up with you.",
        "fr": "J'ai pris votre demande de certification et je l'ai transmise à un collègue, qui vous répondra.",
        "es": "He tomado su solicitud de certificación y la he pasado a un colega, que se pondrá en contacto con usted.",
    },
    "not_in_catalogue": {
        "en": "I can't find that boot in our product list, so I can't give an answer. I've passed your request to a colleague.",
        "fr": "Je ne trouve pas cette botte dans notre liste de produits, alors je ne peux pas répondre. J'ai transmis votre demande à un collègue.",
        "es": "No encuentro esa bota en nuestra lista de productos, así que no puedo responder. He pasado su solicitud a un colega.",
    },
}
SOFT_NOTE = {
    "en": "This reflects the rules and data provided; the final decision rests with customs.",
    "fr": "Ceci reflète les règles et les données fournies; la décision finale revient aux douanes.",
    "es": "Esto refleja las reglas y los datos proporcionados; la decisión final corresponde a la aduana.",
}
DRAFT_LABEL = {"en": "DRAFT, FOR SIGNATURE", "fr": "BROUILLON, À SIGNER", "es": "BORRADOR, PARA FIRMA"}
BASIS_LABEL = {"en": "Basis:", "fr": "Base :", "es": "Base:"}
DUTY_NOTE = {
    "en": "Qualifying does not mean duty-free. Additional tariffs still apply:",
    "fr": "Être admissible ne veut pas dire exempt de droits. Des tarifs additionnels s'appliquent encore :",
    "es": "Calificar no significa estar libre de aranceles. Aún se aplican aranceles adicionales:",
}


# --------------------------------------------------------------------------- results
@dataclass
class Finding:
    rule_id: str
    action: str                      # "fix" or "block"
    detail: str
    reply_key: str = "handoff"       # which scripted reply a block uses
    earlier_layer_failed: bool = False


@dataclass
class ReviewResult:
    decision: str                    # "pass" | "fix" | "hand_to_person"
    answer: str                      # text to send (fixed text, original, or scripted reply)
    findings: list[Finding] = field(default_factory=list)
    language: str = "en"
    rescued: bool = False            # an earlier layer failed and the reviewer caught it
    escalation: dict | None = None   # for create_escalation(reason, summary, evidence)
    layer: str = LAYER

    @property
    def rule_ids(self) -> list[str]:
        return [f.rule_id for f in self.findings]


# ------------------------------------------------------------------------- utilities
def _fold(text: str) -> str:
    t = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn").replace("\u2019", "'")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p for p in parts if p.strip()]


_EN = {"the", "and", "is", "are", "this", "that", "for", "with", "you", "your", "we", "our", "not", "can", "it",
       "of", "to", "boot", "boots", "does", "under", "please", "which", "have", "has", "will"}
_FR = {"le", "les", "est", "pour", "des", "une", "nous", "vous", "pas", "sur", "avec", "botte", "bottes", "cette",
       "votre", "au", "aux", "du", "qu'est", "ce", "je", "peux", "ne", "sont", "ou", "mais"}
_ES = {"el", "los", "las", "para", "una", "con", "por", "es", "esta", "del", "su", "botas", "bota", "esto", "se",
       "un", "y", "puedo", "no", "son", "muy", "pero"}


def detect_language(text: str) -> str | None:
    """Rough EN/FR/ES guess from common words. None when the text is too short to tell."""
    words = re.findall(r"[a-zà-ÿ']+", text.casefold())
    scores = {"en": sum(w in _EN for w in words), "fr": sum(w in _FR for w in words), "es": sum(w in _ES for w in words)}
    best = max(scores, key=scores.get)
    if scores[best] < 2 or sorted(scores.values())[-2] == scores[best]:
        return None
    return best


# Identifiers whose digits are not figures: boot IDs, POs, dates, HS codes, years, section numbers.
_ID_PATTERNS = [
    r"\b[A-Z]{2,4}-\d[\w-]*",                      # NB-0412-V1, PO-8005, SUP-010, INV-4417, RC-110
    r"\b\d{4}-\d{2}-\d{2}\b",                      # dates
    r"\b\d{4}\.\d{2}(?:\.\d{2}){0,2}\b",           # 6401.99.10
    r"\b\d{4}-\d{4}\b",                            # 6401-6405
    r"\b64\d{2}\b",                                # bare HS headings
    r"\b20\d{2}\b",                                # years
    r"(?i:\b(?:section|article|sección|seccion)\s+\d+\b)",
]
_ID_RE = re.compile("|".join(_ID_PATTERNS))
_NUM_RE = re.compile(
    r"(?P<cur>US\$|CA\$|C\$|\$|CAD ?|USD ?)?"
    r"(?P<num>\d{1,3}(?:[,\u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    r"(?P<pct>\s?%)?"
    r"(?P<cur2>\s?(?:\$|CAD\b|USD\b|dollars?\b))?"
)


@dataclass
class Num:
    value: float
    decimals: int
    kind: str          # "pct" | "money" | "plain"
    start: int
    end: int


def _parse_number(s: str) -> tuple[float, int]:
    s = s.replace("\u00a0", "").replace("\u202f", "")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"\d+,\d{1,2}", s):
        s = s.replace(",", ".")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return float(s), decimals


def _numbers(text: str) -> list[Num]:
    id_spans = [m.span() for m in _ID_RE.finditer(text)]
    out: list[Num] = []
    for m in _NUM_RE.finditer(text):
        if any(m.start() < e and m.end() > s for s, e in id_spans):
            continue
        # skip list numbering such as "1. " at the start of a line
        line_start = text.rfind("\n", 0, m.start()) + 1
        if m.group("num").isdigit() and not text[line_start:m.start()].strip() and re.match(r"[.)]\s", text[m.end():m.end() + 2]):
            continue
        value, dec = _parse_number(m.group("num"))
        kind = "pct" if m.group("pct") else ("money" if (m.group("cur") or m.group("cur2")) else "plain")
        out.append(Num(value, dec, kind, m.start(), m.end()))
    return out


_CONF_KEY = re.compile(r"(?i)(?:^|_)(?:cost|costs|cost_cad|unit_cost|supplier_price|labou?r|labou?r_cad|overhead|overhead_cad|margin|net_cost)(?:_|$)")


@dataclass
class TraceFacts:
    numbers: list[float] = field(default_factory=list)          # allowed numeric values (structured + text)
    conf_values: list[float] = field(default_factory=list)      # confidential numeric values
    verdicts: set[str] = field(default_factory=set)
    boot_ids_ok: set[str] = field(default_factory=set)
    boot_ids_not_found: set[str] = field(default_factory=set)
    supplier_names: set[str] = field(default_factory=set)
    tools: set[str] = field(default_factory=set)
    steps: list[str] = field(default_factory=list)              # human-readable rule steps from tools
    rules_version: str | None = None
    duty_results: list[dict] = field(default_factory=list)
    has_draft_certification: bool = False


def _walk(obj: Any, facts: TraceFacts, key: str, restricted: bool) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, facts, str(k), restricted)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, facts, key, restricted)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        if _CONF_KEY.search(key):
            facts.conf_values.append(float(obj))
            if not restricted:
                facts.numbers.append(float(obj))
        else:
            facts.numbers.append(float(obj))
    elif isinstance(obj, str):
        if key == "verdict":
            facts.verdicts.add(obj)
        if key in ("supplier_name", "supplier") and obj:
            facts.supplier_names.add(obj)
        # numbers inside rule-step text: percentages always allowed, other figures only for cost-cleared tiers
        for n in _numbers(obj):
            if n.kind == "pct" or not restricted:
                facts.numbers.append(n.value)


def _collect(trace: Any, restricted: bool) -> TraceFacts:
    facts = TraceFacts()
    calls = trace if isinstance(trace, (list, tuple)) else ([trace] if trace else [])
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool") or call.get("name") or ""
        if tool:
            facts.tools.add(tool)
        if tool == "draft_certification":
            facts.has_draft_certification = True
        result = call.get("result", call)   # accept either {tool, args, result} or a bare envelope
        args = call.get("args") or {}
        if isinstance(result, dict):
            if result.get("rules_version"):
                facts.rules_version = result["rules_version"]
            for s in result.get("trace") or []:
                if isinstance(s, str):
                    facts.steps.append(s)
            err = result.get("error")
            if isinstance(err, dict) and err.get("code") == "NOT_FOUND":
                for m in re.finditer(r"NB-\d{4}(?:-V\d)?", f"{err.get('message', '')} {args.get('boot_id') or args.get('sku') or ''}"):
                    facts.boot_ids_not_found.add(m.group(0))
            if result.get("ok") is True:
                for m in re.finditer(r"NB-\d{4}(?:-V\d)?", f"{args.get('boot_id') or args.get('sku') or ''} {result.get('data', '')}"):
                    facts.boot_ids_ok.add(m.group(0))
            data = result.get("data")
            if isinstance(data, dict) and (tool in ("duty", "import_duty") or any(re.search(r"special|additional", k) for k in data)):
                facts.duty_results.append(data)
        _walk(args, facts, "", restricted)
        _walk(result, facts, "", restricted)
    return facts


_COST_WORDS = re.compile(r"(?i)\b(?:net cost|costs?|labou?r|overhead|margin|price)\b")


def _basis_steps(steps, hide_costs, hide_names, supplier_names, facts):
    """Rule steps that are safe to quote as the basis of an answer.
    Tiers that cannot see costs only get steps whose figures are percentages and that name no cost.
    Tiers that cannot see supplier names only get steps that name no supplier."""
    names = {n.lower() for n in list(supplier_names or []) + list(facts.supplier_names) if n}
    out = []
    for st in steps:
        if hide_costs and (_COST_WORDS.search(st) or any(n.kind != "pct" for n in _numbers(st))):
            continue
        if hide_names and (re.search(r"\bSUP-\d+\b", st) or any(nm in st.lower() for nm in names)):
            continue
        out.append(st)
    return out


def _matches(n: Num, candidates: Iterable[float]) -> bool:
    tol = 0.5 * 10 ** (-n.decimals) + 1e-9
    for t in candidates:
        if abs(n.value - t) <= tol:
            return True
        if n.kind == "pct" and abs(n.value - t * 100) <= tol:
            return True
    return False


# --------------------------------------------------------------------------- patterns
_GUARANTEE = re.compile(
    r"\bguarantee[sd]?\b|\bwill (?:definitely |certainly |surely )?(?:qualify|be accepted|be approved|pass)\b"
    r"|\bcustoms (?:will|won't) (?:accept|challenge|question|reject)\b|\b(?:no|zero) risk\b|\b100 ?% (?:safe|sure|certain)\b"
    r"|\bi promise\b|\bpromise\b|\bgaranti\w*|\bsera (?:accepte|approuve|admissible)\b|\bva (?:certainement )?(?:se )?qualifier\b"
    r"|\bje (?:te |vous )?promets\b|\bsans (?:aucun )?risque\b|\bgarantiz\w+|\bgarantiza\w*"
    r"|\bsera (?:aceptad\w+|aprobad\w+)\b|\bte lo prometo\b|\bsin riesgo\b")
_GUARANTEE_NEG = re.compile(
    r"\b(?:can't|cannot|can not|unable to|do not|don't|not|no|never|won't)\b.{0,20}\b(?:guarantee|promise)"
    r"|\bno guarantee|\bnot guaranteed|\bne (?:peux|puis|pouvons) pas\b.{0,20}garanti|\bpas de garantie"
    r"|\bno (?:puedo|podemos) garantizar|\bno (?:se )?garantiza|\bsin garant")
_DUTY_FREE = re.compile(
    r"\bduty[- ]free\b|\bno duty\b|\bzero duty\b|\bpays? no (?:duty|tariff)\b|\bexempt from (?:duty|tariff)\w*"
    r"|\bsans droits?\b|\bexempt\w* de droits\b|\blibre de droits\b|\bsin aranceles\b|\bexento de arancel\w*|\blibre de aranceles\b")
_DUTY_FREE_NEG = re.compile(
    r"\b(?:not|n't|never|doesn't|isn't|pas|no necesariamente)\b.{0,40}(?:duty[- ]free|no duty|droits|aranceles)"
    r"|(?:duty[- ]free|no duty).{0,30}\b(?:not guaranteed|may still)")
_SIGN_SEND = re.compile(
    r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:already\s+)?(?:signed|sent|emailed|submitted|filed|e-signed)\b"
    r"|\b(?:has|have) been (?:signed|sent|emailed|submitted|filed)\b"
    r"|\bj'ai (?:signe|envoye|soumis|depose)\b|\bha sido (?:firmad\w+|enviad\w+)\b|\b(?:he|hemos) (?:firmado|enviado)\b")
_SIGN_SEND_OK = re.compile(r"\b(?:request|case|escalation|summary|question|colleague|team member|compliance|demande|collegue|solicitud|colega)\b")
_CERT_TEXT = re.compile(r"\bhereby certif\w+|\bje certifie\b|\bcertifico que\b")
_DRAFT_LABEL = re.compile(r"\bdraft\b|\bbrouillon\b|\bborrador\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-])?(?:\(?\d{3}\)?[\s.-])\d{3}[\s.-]\d{4}(?!\d)")
_VERDICT_NO = re.compile(
    r"\b(?:does not|doesn't|do not|did not|fails? to|will not|won't|not)\s+(?:currently\s+)?qualif(?:y|ies)\b"
    r"|\bn'est pas admissible\b|\bne passe pas\b|\bne se qualifie pas\b|\bno califica\b|\bno cumple\b")
_VERDICT_YES = re.compile(r"\bqualifies\b|\bis (?:eligible|admissible)\b|\best admissible\b|\bcalifica\b")
_VERDICT_REVIEW = re.compile(r"\bneeds[ _]review\b|\bneeds a person\b|\ba person (?:decides|must decide|should decide)\b")
_NOT_ASSERTION = re.compile(r"\?|\b(?:whether|if|once|until|which|confirm|savoir si|laquelle|cual|si)\b")
_BASIS = re.compile(r"\b(?:rule|rules|tariff shift|regional value|rvc|based on|according to|rules engine|selon|d'apres|regle|segun|basado|regla)\w*")
_RUDE = re.compile(r"\b(?:stupid|idiot|dumb|shut up|moron|useless|stupide|idiot\w*|tonto|estupid\w*)\b")
_TOOL_NAMES = ("get_boot", "get_materials", "get_supplier", "check_origin", "what_if", "import_duty",
               "draft_certification", "record_document", "create_escalation")
_CONFIG_MARKERS = re.compile(r"system prompt|prompt systeme|prompt del sistema|my instructions (?:are|say)|here are my (?:instructions|tools)|tool_contract")
_BOOT_ID = re.compile(r"NB-\d{4}(?:-V\d)?")


# -------------------------------------------------------------------------- the review
def review(
    question: str,
    answer: str,
    trace: Any,
    tier: str | None,
    *,
    language: str | None = None,
    policy: dict | None = None,
    catalogue_ids: Iterable[str] | None = None,
    supplier_names: Iterable[str] | None = None,
    allow_small_integers: bool = True,
    llm_review: Callable[[str, str, str, list[Finding]], dict | None] | None = None,
) -> ReviewResult:
    """Check one reply. See the module docstring for the four inputs and the checks.

    language       the user's language when the app knows it (region codes such as "fr-CA" are accepted);
                   otherwise guessed from the question. If the app reviews the English reply before translating
                   it back, pass question=<English processing text> and language="en"
    catalogue_ids  boot IDs that exist (for example from skus.csv). Optional; tool results are also used
    supplier_names supplier names to mask for tier C (for example load_supplier_names('suppliers.csv'))
    llm_review     optional hook (question, answer, tier, findings) -> {"action": "pass"|"fix"|"block", "answer": str, "reason": str}
    """
    pol = policy or DEFAULT_POLICY
    tier = tier if tier in pol["tiers"] else pol.get("unknown_user_tier", UNKNOWN_TIER)
    rules = pol["tiers"][tier]
    hide_costs = rules.get("costs") == "hide"
    hide_names = rules.get("supplier_names") == "hide"

    q = question or ""
    text = answer or ""
    q_lang = _base_language(language) if language else detect_language(q)
    lang = q_lang if q_lang in ("en", "fr", "es") else (detect_language(text) or "en")
    facts = _collect(trace, restricted=hide_costs)
    findings: list[Finding] = []

    def block(rule_id: str, detail: str, reply_key: str = "handoff", failed: bool = False) -> None:
        findings.append(Finding(rule_id, "block", detail, reply_key, failed))

    def fix(rule_id: str, detail: str, failed: bool = False) -> None:
        findings.append(Finding(rule_id, "fix", detail, "handoff", failed))

    # A shared compliance mailbox may be shown only when the
    # tool trace proves that the request was handed to the
    # compliance-contact workflow.
    approved_compliance_handoff = any(
        isinstance(step, dict)
        and step.get("tool") == "create_escalation"
        and (
            (
                step.get("args") or {}
            ).get("handoff_to")
            == "compliance_contact"
            or (
                (
                    (
                        step.get("result") or {}
                    ).get("data") or {}
                ).get("handoff_to")
                == "compliance_contact"
            )
        )
        for step in (
            trace
            if isinstance(trace, (list, tuple))
            else ([trace] if trace else [])
        )
    )

    # 1. configuration leak
    folded = _fold(text)
    tools_named = {t for t in _TOOL_NAMES if t in folded}
    if _CONFIG_MARKERS.search(folded) or len(tools_named) >= 2:
        block("no_config_leak", "reply describes the system prompt or the tool list", "refuse", True)

    # 2. signing, filing or sending
    for s in _sentences(text):
        fs = _fold(s)
        if _SIGN_SEND.search(fs) and not _SIGN_SEND_OK.search(fs):
            block("no_signing_or_sending", "reply claims to have signed, filed or sent something", "sign_send", True)
            break

    # 3. certifications: label drafts, and no draft for tier C
    drafted = facts.has_draft_certification or bool(_CERT_TEXT.search(folded))
    if drafted and tier == "C":
        block("certification_rules", "tier C must not receive a certification draft", "cert_tier_c", bool(_CERT_TEXT.search(folded)))
    elif drafted and not (_DRAFT_LABEL.search(folded) and re.search(r"signature|signer|firma", folded)):
        text = f"{DRAFT_LABEL[lang]}\n{text}"
        fix("certification_rules", "added the DRAFT, FOR SIGNATURE label", True)

    # 4. personal data
    #
    # The fictional shared compliance mailbox is permitted only
    # when the tool trace confirms a compliance-contact escalation.
    # All other email addresses and phone numbers continue to use
    # the existing masking and blocking rules.
    email_contacts = {
        match.group(0)
        for match in _EMAIL.finditer(text)
    }

    phone_contacts = {
        match.group(0)
        for match in _PHONE.finditer(text)
    }

    allowed_emails = set()

    if approved_compliance_handoff:
        allowed_emails.add(
            APPROVED_COMPLIANCE_EMAIL.lower()
        )

    unapproved_emails = {
        email
        for email in email_contacts
        if email.lower() not in allowed_emails
    }

    unapproved_contacts = (
        unapproved_emails
        | phone_contacts
    )

    if len(unapproved_contacts) >= 2:
        block(
            "personal_data",
            "bulk list of personal contact details",
            "confidential",
            True,
        )

    elif unapproved_contacts and tier == "C":
        text = _PHONE.sub(
            MASK_PERSONAL,
            text,
        )

        text = _EMAIL.sub(
            lambda match: (
                match.group(0)
                if match.group(0).lower()
                in allowed_emails
                else MASK_PERSONAL
            ),
            text,
        )

        fix(
            "personal_data",
            "masked a personal contact detail for tier C",
            True,
        )

    # 5. confidential figures and supplier names
    if hide_costs:
        redacted = text
        hit = False
        for n in sorted(_numbers(text), key=lambda x: -x.start):
            is_conf = any(
                abs(n.value - c) <= 0.5 * 10 ** (-n.decimals) + 1e-9 and (n.decimals >= 1 or n.kind != "plain" or n.value == c)
                for c in facts.conf_values
            )
            if is_conf:
                redacted = redacted[:n.start] + MASK_COST + redacted[n.end:]
                hit = True
        if hit:
            text = f"{redacted}\n{_cost_note(lang)}"
            fix("masking_costs", "masked supplier prices or costs this tier cannot see", True)
    if hide_names:
        names = {n for n in list(supplier_names or []) + list(facts.supplier_names) if n}
        redacted = text
        hit = False
        for nm in sorted(names, key=len, reverse=True):
            new = re.sub(re.escape(nm), "a supplier", redacted, flags=re.I)
            hit = hit or new != redacted
            redacted = new
        new = re.sub(r"\bSUP-\d+\b", "a supplier", redacted)
        hit = hit or new != redacted
        if hit:
            text = new
            fix("masking_supplier_names", "masked supplier names or IDs for tier C", True)

    # 6. guarantees and promises
    kept, dropped = [], False
    for s in _sentences(text):
        fs = _fold(s)
        if _GUARANTEE.search(fs) and not _GUARANTEE_NEG.search(fs):
            dropped = True
        else:
            kept.append(s)
    if dropped:
        text = " ".join(kept + [SOFT_NOTE[lang]])
        fix("no_guarantee", "removed a promise or guarantee", True)

    # 7. qualifying is not duty-free
    claim_sentences = [s for s in _sentences(text) if _DUTY_FREE.search(_fold(s)) and not _DUTY_FREE_NEG.search(_fold(s))]
    if claim_sentences:
        extras = [e for d in facts.duty_results for k, v in d.items() if re.search(r"special|additional", k) and v for e in (v if isinstance(v, list) else [v])]
        payable = [d.get(k) for d in facts.duty_results for k in ("duty_if_qualifying_usd", "duty_payable_usd") if isinstance(d.get(k), (int, float))]
        if not facts.duty_results:
            block("qualifying_not_duty_free", "duty-free claim with no duty result behind it", "handoff", True)
        elif any(p > 0 for p in payable):
            block("qualifying_not_duty_free", "duty-free claim, but the duty tool shows duty payable", "handoff", True)
        elif extras:
            text = " ".join([s for s in _sentences(text) if s not in claim_sentences] + [f"{DUTY_NOTE[lang]} {_summarise_tariffs(extras)}."])
            fix("qualifying_not_duty_free", "replaced a duty-free claim with the additional tariffs from the duty tool", True)

    # 8. tone
    if _RUDE.search(_fold(text)):
        text = " ".join(s for s in _sentences(text) if not _RUDE.search(_fold(s)))
        fix("polite_tone", "removed rude wording", True)

    # 9. catalogue only
    ids = set(_BOOT_ID.findall(text))
    unknown = (ids & facts.boot_ids_not_found) | ({i for i in ids if i not in set(catalogue_ids)} if catalogue_ids is not None else set())
    figures = [n for n in _numbers(text) if n.kind in ("pct", "money")]
    asserted = _asserted_verdicts(text)

    # In a verified compliance handoff, wording such as
    # "needs review" describes the human escalation. It is not a
    # qualification verdict about a product.
    if approved_compliance_handoff:
        asserted.discard(
            "needs_review"
        )

    if unknown and (asserted or figures or _VERDICT_REVIEW.search(_fold(text))):
        block("catalogue_only", f"verdict or figures for a boot the tools do not know: {', '.join(sorted(unknown))}", "not_in_catalogue", False)

    # 10. verdict matches the rules engine
    if asserted:
        if not facts.verdicts:
            block("verdict_matches_trace", "verdict given with no rules-engine result behind it")
        elif not asserted <= facts.verdicts:
            block("verdict_matches_trace", f"reply says {sorted(asserted)} but the tools returned {sorted(facts.verdicts)}")

    # 11. figures match the tools
    bad = []
    for n in _numbers(text):
        if allow_small_integers and n.kind == "plain" and n.decimals == 0 and 0 <= n.value <= 10:
            continue
        if _matches(n, facts.numbers) or _matches_question(n, q):
            continue
        bad.append(text[n.start:n.end].strip())
    if bad:
        block("figures_match_trace", f"figures not found in any tool result: {', '.join(bad[:5])}")

    # 12. language
    a_lang = detect_language(text)
    if q_lang in ("en", "fr", "es") and a_lang and a_lang != q_lang:
        block("language_matches", f"user wrote {q_lang} but the reply is in {a_lang}")

    # 13. basis stated (a safe repair: quote the tools' own rule steps, minus anything this tier cannot see)
    basis_steps = _basis_steps(facts.steps, hide_costs, hide_names, supplier_names, facts)
    if (asserted or [n for n in _numbers(text) if n.kind in ("pct", "money")]) and not _BASIS.search(_fold(text)) and basis_steps:
        basis = " ".join(basis_steps[:2])
        if facts.rules_version:
            basis += f" (rules version {facts.rules_version})"
        text = f"{text}\n{BASIS_LABEL[lang]} {basis}"
        fix("basis_stated", "added the rule steps the tools returned")

    # 14. optional model-based review
    if llm_review is not None:
        verdict = llm_review(q, text, tier, list(findings)) or {}
        action = verdict.get("action", "pass")
        if action == "block":
            block("llm_review", str(verdict.get("reason", "model review blocked the reply")))
        elif action == "fix" and verdict.get("answer"):
            text = verdict["answer"]
            fix("llm_review", str(verdict.get("reason", "model review edited the reply")))

    # ---- decide
    blocking = [f for f in findings if f.action == "block"]
    rescued = any(f.earlier_layer_failed for f in findings)
    if blocking:
        reply = REPLIES[blocking[0].reply_key][lang]
        return ReviewResult("hand_to_person", reply, findings, lang, rescued, _escalation(blocking, tier, q))
    if findings:
        return ReviewResult("fix", text, findings, lang, rescued)
    return ReviewResult("pass", answer or "", findings, lang, False)


# ---------------------------------------------------------------------------- helpers
def _base_language(code: str) -> str:
    """'fr-CA' -> 'fr', 'zh-CN' -> 'zh'. The translation service uses region codes."""
    return code.split("-")[0].split("_")[0].lower()


def _cost_note(lang: str) -> str:
    return {"en": "Some figures are confidential and have been masked.",
            "fr": "Certains chiffres sont confidentiels et ont été masqués.",
            "es": "Algunas cifras son confidenciales y se han ocultado."}[lang]


def _summarise_tariffs(extras: list) -> str:
    parts = []
    for e in extras:
        if isinstance(e, dict):
            name = e.get("name") or e.get("tariff") or e.get("code") or "additional tariff"
            rate = e.get("rate")
            parts.append(f"{name} ({rate * 100:g}%)" if isinstance(rate, (int, float)) and 0 < rate <= 1 else str(name))
        else:
            parts.append(str(e))
    return "; ".join(parts)


def _asserted_verdicts(text: str) -> set[str]:
    out: set[str] = set()
    for s in _sentences(text):
        fs = _fold(s)
        if _NOT_ASSERTION.search(fs):
            continue
        no = _VERDICT_NO.search(fs)
        if no:
            out.add("does_not_qualify")
            fs = fs[:no.start()] + " " + fs[no.end():]
        if _VERDICT_YES.search(fs):
            out.add("qualifies")
        if _VERDICT_REVIEW.search(fs):
            out.add("needs_review")
    return out


def _matches_question(n: Num, question: str) -> bool:
    return any(abs(n.value - q.value) <= 1e-9 for q in _numbers(question))


def _escalation(blocking: list[Finding], tier: str, question: str) -> dict:
    first = blocking[0]
    return {
        "reason": first.rule_id,
        "summary": f"Reviewer blocked a reply for a tier {tier} user: {first.detail}",
        "evidence": [{"rule": f.rule_id, "detail": f.detail} for f in blocking],
        "question": question,
    }


if __name__ == "__main__":
    demo = review(
        "Does NB-0412 qualify under CUSMA?",
        "NB-0412 will qualify, guaranteed. The regional value content is 65.3%.",
        [{"tool": "check_origin", "args": {"boot_id": "NB-0412"},
          "result": {"ok": True, "data": {"boot_id": "NB-0412", "verdict": "qualifies", "rvc": {"value": 0.653, "threshold": 0.55}},
                     "trace": ["RVC = 65.3%, above 55%"], "warnings": [], "rules_version": "2026-10-24"}}],
        "A",
    )
    print(demo.decision, "|", demo.answer, "|", demo.rule_ids)
