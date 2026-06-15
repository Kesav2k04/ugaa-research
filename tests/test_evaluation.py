import pytest
import torch
from unittest.mock import MagicMock

from pope_audit import YES_TOKEN_IDS, NO_TOKEN_IDS
from pope_audit.evaluate import compute_f1
from pope_audit.ugaa_hook import _get_yes_no_logits

def test_parser_logit_pooling():
    """
    The Parser Test: Validates that the inference parsing function correctly 
    extracts and pools the 8-token logit indices from the raw model output.
    """
    vocab_size = 32000
    mock_logits_tensor = torch.full((vocab_size,), -5.0)

    # Construct a scenario where a space-prefixed token ID (8241) holds the highest logit
    # value in the vector (i.e. model's true positive intent)
    mock_logits_tensor[8241] = 24.6
    
    # Unprefixed IDs contain low background noise values
    mock_logits_tensor[3582] = -4.2  # rigid 'yes' token
    mock_logits_tensor[1217] = -4.0  # rigid 'no' token (slightly higher noise)
    
    # Fill remaining required IDs with base noise (-6.0)
    for token_id in YES_TOKEN_IDS + NO_TOKEN_IDS:
        if token_id not in [8241, 3582, 1217]:
            mock_logits_tensor[token_id] = -6.0

    # Mock the VLM components
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_out = MagicMock()
    # Simulates a tuple containing a 2D tensor of shape (1, vocab_size)
    # mirroring Hugging Face's return_dict_in_generate=True, output_scores=True structure.
    mock_out.scores = (mock_logits_tensor.unsqueeze(0),)
    mock_model.generate.return_value = mock_out

    mock_processor = MagicMock()
    mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    # Call the exact token-parsing utility function from our codebase
    with torch.no_grad():
        logit_yes, logit_no = _get_yes_no_logits(
            model=mock_model,
            processor=mock_processor,
            image=None,
            question="Is there a snowboard?",
            device="cpu"
        )

    # Assert the rigid 2-token logic fails implicitly because -4.2 < -4.0
    # But assert that our 8-token logit parsing function successfully extracted 
    # the space-prefixed logit and mapped it accurately.
    assert logit_yes == pytest.approx(24.6, abs=1e-4)
    assert logit_no == pytest.approx(-4.0, abs=1e-4)
    assert logit_yes > logit_no


def test_metrics_f1_computation():
    """
    The Metrics Test: compute_f1() should be tested by passing resulting
    high-level classification strings against ground-truth annotations to verify
    the mathematical precision and recall formulas.
    """
    mock_dataset = [
        {"question_id": 1, "question": "Is there a snowboard in the image?", "label": "yes", "split": "adversarial"},
        {"question_id": 2, "question": "Is there a dining table in the image?", "label": "no", "split": "adversarial"},
        {"question_id": 3, "question": "Is there a bus in the image?", "label": "no", "split": "adversarial"},
        {"question_id": 4, "question": "Is there a bed in the image?", "label": "yes", "split": "adversarial"}
    ]

    # Model predictions mapped from max logits logic
    predictions = ["yes", "no", "yes", "no"]
    ground_truths = [item["label"] for item in mock_dataset]

    results = compute_f1(predictions, ground_truths)
    
    assert results["tp"] == 1
    assert results["fp"] == 1
    assert results["fn"] == 1
    assert results["precision"] == 0.5
    assert results["recall"] == 0.5
    assert results["f1"] == 0.5

def test_metrics_accuracy_computation():
    """
    The Metrics Test (Accuracy): compute_accuracy() should be tested for both 
    string and boolean ground-truth normalization since some datasets output True/False.
    """
    from pope_audit.evaluate import compute_accuracy

    predictions = ["yes", "no", "yes", "no", "yes", "no"]
    # Provide mixed string and boolean ground truths to test normalization
    ground_truths = ["yes", False, True, "yes", "no", False]
    
    # Normalized ground truths internally become:
    # ["yes", "no", "yes", "yes", "no", "no"]
    #
    # Matches:
    # pred[0]="yes" == "yes" (Match)
    # pred[1]="no"  == "no"  (Match)
    # pred[2]="yes" == "yes" (Match)
    # pred[3]="no"  != "yes" (Mismatch)
    # pred[4]="yes" != "no"  (Mismatch)
    # pred[5]="no"  == "no"  (Match)
    #
    # Correct: 4, Total: 6 -> Accuracy = 0.6667

    results = compute_accuracy(predictions, ground_truths)
    
    assert results["correct"] == 4
    assert results["total"] == 6
    assert results["accuracy"] == pytest.approx(0.6667, abs=1e-4)

def test_evaluate_predictions_file_io(tmp_path):
    """
    The Integration Test: Validates that evaluate_predictions correctly loads 
    and parses file streams dynamically. Uses pytest tmp_path to securely test 
    IO bounds without cluttering the local repository.
    """
    import json
    from pope_audit.evaluate import evaluate_predictions
    
    # Securely create mock POPE JSONL dataset
    pope_file = tmp_path / "mock_pope.json"
    pope_data = [
        {"image": "001.jpg", "text": "Is there a cat?", "label": "yes"},
        {"image": "002.jpg", "text": "Is there a dog?", "label": "no"}
    ]
    with open(pope_file, "w") as f:
        for item in pope_data:
            f.write(json.dumps(item) + "\n")
            
    # Securely create mock predictions text file
    pred_file = tmp_path / "mock_preds.txt"
    with open(pred_file, "w") as f:
        f.write("yes\nno\n")
        
    results = evaluate_predictions(str(pred_file), str(pope_file))
    
    # Assert end-to-end integration works seamlessly
    assert results["tp"] == 1
    assert results["fp"] == 0
    assert results["fn"] == 0
    assert results["precision"] == 1.0
    assert results["f1"] == 1.0
