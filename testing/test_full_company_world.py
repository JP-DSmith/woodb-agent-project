import csv
from collections import Counter
from pathlib import Path


WORLD_FOLDER = Path(
    "prototypes/cusma_qualification/World"
)

VALID_VERDICTS = {
    "qualifies",
    "needs_review",
    "does_not_qualify",
}


def load_csv(filename):
    """Load a company-world CSV file as dictionaries."""
    path = WORLD_FOLDER / filename

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def test_full_company_world_has_expected_scale():
    """The company world should contain approximately 50 boots."""
    skus = load_csv("skus.csv")
    bom_lines = load_csv("bom_lines.csv")
    suppliers = load_csv("suppliers.csv")
    customers = load_csv("customers.csv")
    orders = load_csv("orders.csv")

    assert len(skus) >= 40
    assert len(bom_lines) >= len(skus)
    assert len(suppliers) > 0
    assert len(customers) > 0
    assert len(orders) == len(skus)


def test_every_sku_has_unique_ground_truth():
    """Every company-world SKU must have exactly one truth record."""
    skus = load_csv("skus.csv")
    ground_truth = load_csv("ground_truth.csv")

    sku_ids = [row["sku"] for row in skus]
    truth_ids = [row["sku"] for row in ground_truth]

    assert len(sku_ids) == len(set(sku_ids))
    assert len(truth_ids) == len(set(truth_ids))
    assert set(sku_ids) == set(truth_ids)


def test_company_world_references_are_valid():
    """BOM, order, customer and supplier references must resolve."""
    skus = load_csv("skus.csv")
    bom_lines = load_csv("bom_lines.csv")
    suppliers = load_csv("suppliers.csv")
    customers = load_csv("customers.csv")
    orders = load_csv("orders.csv")

    sku_ids = {
        row["sku"]
        for row in skus
    }

    supplier_ids = {
        row["supplier_id"]
        for row in suppliers
    }

    customer_ids = {
        row["customer_id"]
        for row in customers
    }

    bom_supplier_ids = {
        row["supplier_id"]
        for row in bom_lines
        if row["supplier_id"].strip()
    }

    order_skus = {
        row["sku"]
        for row in orders
    }

    order_customer_ids = {
        row["customer_id"]
        for row in orders
    }

    assert bom_supplier_ids <= supplier_ids
    assert order_skus <= sku_ids
    assert order_customer_ids <= customer_ids


def test_every_order_has_order_ground_truth():
    """Every order must have exactly one order-level truth record."""
    orders = load_csv("orders.csv")
    order_truth = load_csv("order_truth.csv")

    order_ids = [
        row["po_number"]
        for row in orders
    ]

    truth_ids = [
        row["po_number"]
        for row in order_truth
    ]

    assert len(order_ids) == len(set(order_ids))
    assert len(truth_ids) == len(set(truth_ids))
    assert set(order_ids) == set(truth_ids)


def test_ground_truth_uses_valid_verdicts():
    """Ground-truth verdicts must use the approved vocabulary."""
    ground_truth = load_csv("ground_truth.csv")

    actual_verdicts = {
        row["verdict"]
        for row in ground_truth
    }

    assert actual_verdicts <= VALID_VERDICTS


def test_review_cases_include_explanations():
    """Every human-review decision must explain the escalation."""
    ground_truth = load_csv("ground_truth.csv")

    missing_reasons = [
        row["sku"]
        for row in ground_truth
        if row["verdict"] == "needs_review"
        and not row["reason"].strip()
    ]

    assert missing_reasons == []


def test_blank_hs_codes_are_intentional_review_cases():
    """Blank classifications are valid only for explained review cases."""
    ground_truth = load_csv("ground_truth.csv")

    blank_hs_rows = [
        row
        for row in ground_truth
        if not row["hs_code"].strip()
    ]

    assert len(blank_hs_rows) > 0

    for row in blank_hs_rows:
        assert row["verdict"] == "needs_review"
        assert row["reason"].strip()


def test_company_world_contains_planted_problem_cases():
    """The world must preserve multiple deterministic test outcomes."""
    ground_truth = load_csv("ground_truth.csv")

    verdict_counts = Counter(
        row["verdict"]
        for row in ground_truth
    )

    tariff_shift_failures = sum(
        row["tariff_shift_ok"].strip().lower() == "false"
        for row in ground_truth
    )

    assert verdict_counts["qualifies"] > 0
    assert verdict_counts["needs_review"] > 0
    assert verdict_counts["does_not_qualify"] > 0
    assert tariff_shift_failures > 0