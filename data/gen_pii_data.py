#!/usr/bin/env python3
# generate_pii_training_data_json_random_prompt.py

import json
import random
from datasets import load_dataset

def main(split="train"):
    # 1. Load the train split
    ds = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy", split=split)

    # 2. Extract the full set of labels
    all_labels = sorted({
        span["label"]
        for ex in ds
        for span in ex["privacy_mask"]
    })

    records = []
    for ex in ds:
        text = ex["source_text"]
        spans = ex["privacy_mask"]

        # 3. Randomly pick a subset of labels for this prompt
        #    (we choose between 1 and len(all_labels) labels)
        k = random.randint(1, len(all_labels))
        chosen = random.sample(all_labels, k)

        # 4. Build the system prompt for just those labels
        system_prompt = (
            "You are a PII extraction model. Identify all PII entities in the input text. "
            f"Only look for these types: {', '.join(chosen)}. "
            "Return output as a JSON array of objects, each with exactly 'label' and 'value'."
        )

        # 5. Filter spans to only the chosen labels
        filtered = [
            {"label": span["label"], "value": text[span["start"]:span["end"]]}
            for span in spans
            if span["label"] in chosen
        ]

        # 6. Append record
        records.append({
            "instruction": system_prompt,
            "input": "# PII text\n" + text,
            "output": json.dumps(filtered, ensure_ascii=False)
        })

    # Find the longest input text
    max_input_length = max(len(record["input"]) for record in records)
    print(f"Longest input text length: {max_input_length} characters")

    filename = "pii.json"
    if split == "validation":
        filename = "pii_test.json"

    # 7. Write out a single JSON file
    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(records, fout, ensure_ascii=False, indent=2)

    print(f"✓ Wrote {len(records)} records to {filename}")


if __name__ == "__main__":
    main(split="validation")
