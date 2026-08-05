from pathlib import Path
from synthetic_validation.config import load_config, config_hash

CONFIG_DIR = Path(__file__).resolve().parents[2] / "synthetic_validation" / "configs"

def test_load_and_hash_truth_parent_differ():
    truth = load_config(CONFIG_DIR / "truth.json")
    parent = load_config(CONFIG_DIR / "parent.json")
    assert truth.grid_nx == 29 and truth.base_edge == "bottom"
    # parent is deliberately imperfect -> different modulus -> different hash
    assert parent.E != truth.E
    assert config_hash(truth) != config_hash(parent)
    # hashing is stable
    assert config_hash(truth) == config_hash(load_config(CONFIG_DIR / "truth.json"))
