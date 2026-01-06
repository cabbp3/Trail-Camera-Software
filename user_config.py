"""
User Configuration

Simple username management for Phase 1 user system.
Stores username locally - no authentication yet.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".trailcam" / "user_config.json"


def get_username() -> Optional[str]:
    """Get the stored username, or None if not set."""
    if not CONFIG_PATH.exists():
        return None

    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        return config.get("username")
    except Exception as e:
        logger.error(f"Failed to read user config: {e}")
        return None


def set_username(username: str) -> bool:
    """Store the username."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new
        config = {}
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = json.load(f)

        config["username"] = username

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Username set to: {username}")
        return True
    except Exception as e:
        logger.error(f"Failed to save username: {e}")
        return False


def get_config() -> dict:
    """Get full user config."""
    if not CONFIG_PATH.exists():
        return {}

    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except:
        return {}


def set_config_value(key: str, value) -> bool:
    """Set a config value."""
    try:
        config = get_config()
        config[key] = value

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def prompt_for_username_cli() -> str:
    """CLI prompt for username (for scripts)."""
    print("\n=== Trail Camera Software ===")
    print("Please enter your username.")
    print("This identifies your photos in the cloud.")
    print("(Use something simple like your first name)\n")

    while True:
        username = input("Username: ").strip()
        if username and len(username) >= 2:
            set_username(username)
            print(f"\nWelcome, {username}!")
            return username
        print("Username must be at least 2 characters.")


if __name__ == "__main__":
    # Test
    current = get_username()
    if current:
        print(f"Current username: {current}")
    else:
        prompt_for_username_cli()
