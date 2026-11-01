# T16 Document generator v1 (Kush)
Run:  pip install faker
      python generate_documents.py --world world --templates T10_templates --out output --n 10
--n 10 makes the fixed 10-document sample; any other number draws documents at random (seeded).
Output: output/documents/*.txt, output/ground_truth.json (answer keys + planted problems),
        output/hidden_instructions.json (for Mahek's attack tests).
See WORLD_GAPS.md for what world v1 is missing.
