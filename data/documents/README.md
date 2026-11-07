# Supplier documents (tasks T10, T16, T21)

Synthetic supplier paperwork (invoices, origin declarations, supplier emails, purchase orders) in
English, French and Spanish, with planted problems and an answer key. Input for the intake agent and
the attack tests.

## Run (from the repo root)
    pip install -r requirements.txt
    python data/documents/generate_documents.py                  # 300 documents -> data/documents/output/
    python data/documents/generate_documents.py --n 10 --out data/documents/output_sample10
    python data/documents/check_templates.py                     # after editing any template
    pytest -q testing/test_supplier_documents.py

Same seed (default 21) = identical output for everyone.

## Reads
- prototypes/cusma_qualification/World/suppliers.csv: supplier, country, language, declaration_status, declaration_expiry
- prototypes/cusma_qualification/World/bom_lines.csv: description, hs_code, origin_country, cost_cad, sku, line_no
- templates/ and T10_spec.json in this folder
Not read: orders.csv (customer orders), data/world_v1/ (old world).

## Writes (output/)
- documents/DOC-xxxx_<type>_<lang>.txt
- ground_truth.json: true facts per document; items carry bom_ref (sku#line_no), used_in_skus,
  counts_as_originating (North American origin AND a valid supplier declaration on the document date)
  and excluded_input_6406_10; planted problems (codes M1-U1 from T10_spec.json); world_flags;
  expected_intake_action (record / record_with_note / flag_for_person / flag_security / reject_unsupported)
- hidden_instructions.json: documents containing hidden instructions, for the attack tests

## Stand-ins until the World provides them
- part_no = <supplier_id>-NN, one per supplier product
- supplier address and contact: Faker seeded by supplier_id (same supplier, same address)
- supplier purchase orders: SPO-0001... (never PO-80xx, which are customer orders)
- buyer (Northfield Rubber Footwear Inc.) address and contact: constants in generate_documents.py
