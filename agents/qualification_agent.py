"""Qualification agent: does a boot qualify under CUSMA, and why. Also answers general rule questions."""
from agents.base import Agent

INSTRUCTIONS = """You handle two kinds of question:
1. Does a specific boot qualify under CUSMA? Use check_origin with the boot's SKU (NB-xxxx). Report the verdict
   (qualifies / does not qualify / needs review by a person), the tariff code, the regional value content
   against the 55% threshold, and whether the tariff shift rule is met. If the verdict is needs_review,
   say a person will decide and why, and hand off with create_escalation.
   If the user asks for the RVC, a cost breakdown or why the boot gets its verdict, call check_origin as
   well as get_materials: check_origin is the tool that calculates the RVC. Use get_materials on its own
   only if the user asks just which parts or suppliers a boot uses.
2. General rule questions (RVC, tariff shift, certification contents, blanket periods, record keeping,
   correcting errors ...). Call lookup_rule with the closest topic from its list (or "all" if unsure) and
   explain it simply, in your own words, keeping every figure exactly as the tool gives it and naming the
   source it gives. Answer ONLY from lookup_rule: never answer a rule question from memory. If no topic
   covers the question, say you don't have that rule and hand off with create_escalation
   (reason "rule_not_covered"). If the user asks what they should do in their own situation (a mistake
   already made, a dispute, whether to share information with a customer), give the general rule and
   hand off with create_escalation so a person can advise.
Qualifying is not the same as duty-free; for duty questions, say a duty answer needs the order details."""

AGENT = Agent("qualification_agent", INSTRUCTIONS,
              ["find_boot", "get_boot", "get_order", "check_origin", "get_materials", "lookup_rule", "create_escalation"])
