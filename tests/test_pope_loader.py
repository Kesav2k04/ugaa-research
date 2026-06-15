import pytest
import json
from pope_audit.pope_loader import load_pope_split, get_ground_truths

def test_load_pope_split(tmp_path):
    """
    Validates load_pope_split parses JSONL schema matching the split name.
    """
    # Create file matching the split name 'adversarial'
    test_file = tmp_path / "pope_adversarial_test.json"
    data = [
        {"image": "img1.jpg", "text": "Q1", "label": "yes"},
        {"image": "img2.jpg", "text": "Q2", "answer": "no"}
    ]
    with open(test_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    # Add a dummy file to ensure it picks the right one
    dummy_file = tmp_path / "pope_random_test.json"
    dummy_file.write_text('{"image": "x", "label": "no"}\n')

    loaded_data = load_pope_split(str(tmp_path), "adversarial")
    
    assert len(loaded_data) == 2
    assert loaded_data[0]["image"] == "img1.jpg"

def test_get_ground_truths():
    """
    Validates extracting labels handling both 'label' and 'answer' keys, 
    including case and whitespace normalization.
    """
    data = [
        {"label": "yes"},
        {"answer": " No "}, # Testing stripping and lowering
        {"label": "YES"}
    ]
    
    labels = get_ground_truths(data)
    assert labels == ["yes", "no", "yes"]

def test_get_ground_truths_missing_key():
    """
    Ensure KeyError is raised defensively if schema is invalid.
    """
    data = [{"invalid": "yes"}]
    with pytest.raises(KeyError):
        get_ground_truths(data)

def test_load_pope_split_missing_file(tmp_path):
    """
    Ensure FileNotFoundError is raised if no file matches the target split.
    """
    with pytest.raises(FileNotFoundError):
        load_pope_split(str(tmp_path), "popular")
