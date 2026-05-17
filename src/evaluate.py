
# src/evaluate.py
# KARTHIGEYAN builds this on HP Pavilion
# Goal: build the evaluation harness for POPE benchmark

import json
import os

def load_pope(pope_path):
    """
    Load POPE benchmark questions.
    POPE format: each line is a JSON object with:
    - image: image filename
    - text: question (e.g., "Is there a car in the image?")
    - label: "yes" or "no" (ground truth)
    """
    data = []
    with open(pope_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def compute_f1(predictions, ground_truths):
    """
    Compute F1 score for binary yes/no predictions.
    predictions: list of "yes"/"no" strings
    ground_truths: list of "yes"/"no" strings
    """
    tp = sum(1 for p, g in zip(predictions, ground_truths) 
             if p.lower() == 'yes' and g.lower() == 'yes')
    fp = sum(1 for p, g in zip(predictions, ground_truths) 
             if p.lower() == 'yes' and g.lower() == 'no')
    fn = sum(1 for p, g in zip(predictions, ground_truths) 
             if p.lower() == 'no' and g.lower() == 'yes')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'tp': tp, 'fp': fp, 'fn': fn
    }

def evaluate_predictions(pred_file, pope_file):
    """
    Compare model predictions against POPE ground truth.
    pred_file: path to your model's output (one answer per line)
    pope_file: path to POPE JSON file
    """
    pope_data = load_pope(pope_file)
    ground_truths = [item['label'] for item in pope_data]
    
    with open(pred_file, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]
    
    assert len(predictions) == len(ground_truths), \
        f"Mismatch: {len(predictions)} preds vs {len(ground_truths)} labels"
    
    results = compute_f1(predictions, ground_truths)
    print(f"Results: {results}")
    return results

# Test with dummy data to verify the logic works
# To this:
if __name__ == "__main__":

    dummy_preds  = ["yes", "no", "yes", "yes", "no"]
    dummy_labels = ["yes", "no", "no",  "yes", "yes"]
    result = compute_f1(dummy_preds, dummy_labels)
    print("Dummy test result:", result)
    # Expected: precision=0.6667, recall=0.6667, f1=0.6667