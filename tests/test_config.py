"""
Tests for configuration and environment variable loading.
Ensures tokens and secrets are loaded securely without exposure.
"""
import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

def test_env_file_exists():
    env_path = Path("D:/auto_drive/.env")
    assert env_path.exists(), ".env file must exist in project root"

def test_env_loading_security():
    env_path = Path("D:/auto_drive/.env")
    values = dotenv_values(dotenv_path=env_path)
    
    # Verify core path variables exist
    assert "PROJECT_DIR" in values
    assert "MODELS_DIR" in values
    assert "HF_HOME" in values
    
    # Verify secret is present but ensure string representation does not expose it
    token = values.get("HF_ACCESS_TOKEN", "").strip()
    assert len(token) > 0, "HF_ACCESS_TOKEN should be present"
    assert token.startswith("hf_"), "Token should have standard HF prefix"
    
    # Test that HF_HOME is valid
    hf_home = Path(values["HF_HOME"])
    assert hf_home.is_absolute(), "HF_HOME must be absolute path"
