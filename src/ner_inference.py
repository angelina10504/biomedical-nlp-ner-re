
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

def load_ner_model(model_path):
    label_names = ["O", "B-Disease", "I-Disease"]
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, label_names, device

def predict_entities(text, model, tokenizer, label_names, device):
    words = text.split()
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    word_ids = tokenized.word_ids()

    word_predictions = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_predictions:
            word_predictions[word_id] = label_names[predictions[idx]]

    entities = []
    current_entity = []

    for word_id, label in sorted(word_predictions.items()):
        word = words[word_id]
        if label == "B-Disease":
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = [word]
        elif label == "I-Disease" and current_entity:
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return {"text": text, "entities": entities}
