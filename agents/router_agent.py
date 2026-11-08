import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Descriptive Agent Mapping following design specifications
AGENT_MAPPING = {
    "qualification_check": "Qualification Agent (3a_qualification_agent)",
    "rules_info": "Qualification Agent (3a_qualification_agent)",
    "sourcing_whatif": "Sourcing Agent (3b_sourcing_agent)",
    "duty_cost": "Duty and Cost Agent (3c_duty_cost_agent)",
    "certification_draft": "Certification Agent (3d_certification_agent)",
    "document_intake": "Intake Agent (1_intake_agent)",
    "escalate_human": "Human Escalate (human)",
    "out_of_scope": "Polite Decline (polite_decline)"
}

# Initialize model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = embedding_model.to(device)

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return embeddings.cpu().numpy()

# Precompute references from official training sources on import
reference_texts = []
reference_labels = []

# 1. Load request types and exemplars from config
config_path = 'config/request_types_and_personas.json'
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    for rt in config_data.get('request_types', []):
        label_id = rt['id']
        examples = rt.get('examples', {})
        for ex in examples.get('en', []) + examples.get('fr', []):
            reference_texts.append(ex)
            reference_labels.append(label_id)

# 2. Add queries from train_dataset.jsonl
train_path = 'data/train_dataset.jsonl'
if os.path.exists(train_path):
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                reference_texts.append(item['query'])
                reference_labels.append(item['intent'])

# Precompute reference embeddings
if reference_texts:
    reference_embeddings = get_embeddings(reference_texts)
else:
    reference_embeddings = np.array([])

def route_message_few_shot(message: str):
    if len(reference_embeddings) == 0:
        return {
            "message": message,
            "predicted_class": "escalate_human",
            "target_agent": AGENT_MAPPING.get("escalate_human"),
            "similarity_score": 0.0,
            "matched_example": ""
        }

    query_emb = get_embeddings([message])
    dot_product = np.dot(query_emb, reference_embeddings.T)
    query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    ref_norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True).T
    similarities = dot_product / (query_norm * ref_norms)

    best_idx = np.argmax(similarities[0])
    best_score = similarities[0][best_idx]
    best_label = reference_labels[best_idx]

    return {
        "message": message,
        "predicted_class": best_label,
        "target_agent": AGENT_MAPPING.get(best_label, "Unknown"),
        "similarity_score": round(float(best_score), 4),
        "matched_example": reference_texts[best_idx]
    }

def route_user_input(message: str):
    return route_message_few_shot(message)
