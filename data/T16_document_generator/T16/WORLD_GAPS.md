# T16 notes: gaps in World v1 (T13) - for Daniel

The generator works with world v1. These details were missing, so T16 fills them in itself
(in PARTS_BY_SUPPLIER / constants at the top of generate_documents.py). For T21 (full ~300 docs),
it would be better if the world supplied them:

1. Part numbers, units and unit prices per supplier material (T16 invents 1-2 parts per supplier).
2. HS codes per material. T16 uses heading-level placeholders - UNVERIFIED, replace with Nazia's T06 codes.
3. Supplier addresses and contact people (T16 uses Faker).
4. Which suppliers supply which boots (a materials list / bill of materials per boot).
5. A Quebec supplier. There is none, so T16 assumes SUP-01 Northern Tanning (CA) is in Quebec to get French docs.
6. Purchase orders to suppliers. orders.json holds customer sales orders; T16 reuses order IDs as PO references.
7. Boot styles: world v1 has leather work/hiking/dress boots, but the plan and Kush's guide list 7 rubber
   boot styles (knee boot, steel-toe rubber, insulated TPU, ankle rubber, pac boot, neoprene, import-heavy).
   Please confirm which is right before T20.

Known v1 limits of T16: material descriptions stay in English inside French/Spanish documents;
defects C1, C3, C4, M4, H3, T4, U1 from the catalogue are not implemented yet (planned for T21).
