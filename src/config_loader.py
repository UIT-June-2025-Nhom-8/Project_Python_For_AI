import json


def load_json_config(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from: {config_file}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Configuration file {config_file} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {config_file}: {e}")
        return None
