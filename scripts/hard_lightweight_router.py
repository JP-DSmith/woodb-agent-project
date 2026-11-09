from test_lightweight_router import route


HARD = [
    (
        "How much duty will Lakehead Workwear pay on PO-8029?",
        "duty_cost",
    ),
    (
        "NB-0106 qualifies for CUSMA, so my customer pays no duty on it, right?",
        "duty_cost",
    ),
    (
        "Why is the women's Tundra Pac leather boot charged extra "
        "when men's leather boots aren't?",
        "rules_info",
    ),
    (
        "NB-0413 is borderline. If we buy its outsole from a Mexican "
        "supplier instead of the one in Vietnam, does it qualify?",
        "sourcing_whatif",
    ),
    (
        "Can we get NB-0414 to qualify just by changing suppliers?",
        "sourcing_whatif",
    ),
    (
        "Would switching the labels on NB-0413 to a Canadian supplier "
        "be enough to make it qualify?",
        "sourcing_whatif",
    ),
    (
        "Does NB-0412-V3 qualify under CUSMA?",
        "qualification_check",
    ),
    (
        "Why does the height of a rubber boot matter for CUSMA?",
        "rules_info",
    ),
    (
        "NB-0102 is 15 inches tall. Doesn't that put it in your "
        "knee review band?",
        "qualification_check",
    ),
    (
        "Compare the duty on PO-8030 and PO-8031.",
        "duty_cost",
    ),
    (
        "C'est vrai que les bottes en cuir pour femmes ont une taxe "
        "de plus aux États-Unis? Pourquoi?",
        "rules_info",
    ),
    (
        "Est-ce que la botte d'hiver PAC-40 est admissible à l'ACEUM?",
        "qualification_check",
    ),
    (
        "Si on achète les semelles au Québec plutôt qu'en Chine, "
        "qu'est-ce qui change?",
        "sourcing_whatif",
    ),
    (
        "Quels droits de douane payons-nous sur les doublures "
        "importées d'Asie?",
        "duty_cost",
    ),
    (
        "Prépare un certificat d'origine pour la commande du client "
        "à Boston.",
        "certification_draft",
    ),
    (
        "Voici la déclaration du fournisseur, peux-tu vérifier l'origine?",
        "document_intake",
    ),
    (
        "Le client conteste l'origine de nos bottes, on fait quoi légalement?",
        "escalate_human",
    ),
    (
        "Peux-tu m'aider avec ma déclaration d'impôt?",
        "out_of_scope",
    ),
]


correct = 0
mistakes = []


for question, expected in HARD:

    result = route(question)

    predicted = result[
        "predicted_class"
    ]

    score = result["score"]

    passed = predicted == expected

    correct += passed

    if not passed:
        mistakes.append(
            (
                expected,
                predicted,
                score,
                question,
            )
        )

    print(
        f"{'PASS' if passed else 'FAIL':4s} "
        f"{predicted:20s} "
        f"(wanted {expected:20s}) "
        f"score {score:.2f} | "
        f"{question[:80]}"
    )


accuracy = correct / len(HARD)

print()
print("=" * 70)

print(
    f"Overall: {correct}/{len(HARD)} "
    f"= {accuracy:.0%}"
)

print()
print("Mistakes:")

if not mistakes:
    print("None")

else:
    for (
        expected,
        predicted,
        score,
        question,
    ) in mistakes:

        print(
            f"wanted {expected:20s} "
            f"got {predicted:20s} "
            f"score {score:.2f} | "
            f"{question}"
        )


print()
print("=" * 70)

if correct < 14:
    raise SystemExit(
        "FAIL: Lightweight router performed "
        "worse than the PyTorch baseline."
    )

print(
    "PASS: Lightweight router matches or "
    "beats the previous hard-test baseline."
)
