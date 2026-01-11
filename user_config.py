"""
User Configuration

Simple username and hunting club management.
Stores settings locally - no authentication yet.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".trailcam" / "user_config.json"

# Default clubs (matches collections in database)
DEFAULT_CLUBS = [
    "Brooke Farm",
    "Tightwad House",
    "Hunting Club",
]


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


def get_hunting_clubs() -> List[str]:
    """Get the stored hunting clubs (user can be in multiple)."""
    if not CONFIG_PATH.exists():
        return []

    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        clubs = config.get("hunting_clubs", [])
        # Handle legacy single club format
        if not clubs and config.get("hunting_club"):
            clubs = [config.get("hunting_club")]
        return clubs
    except Exception as e:
        logger.error(f"Failed to read user config: {e}")
        return []


def get_hunting_club() -> Optional[str]:
    """Get the first hunting club (for backwards compatibility)."""
    clubs = get_hunting_clubs()
    return clubs[0] if clubs else None


def set_hunting_clubs(clubs: List[str]) -> bool:
    """Store the hunting clubs (multiple allowed)."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        config = {}
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = json.load(f)

        config["hunting_clubs"] = clubs
        # Clear legacy single club field
        config.pop("hunting_club", None)

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Hunting clubs set to: {clubs}")
        return True
    except Exception as e:
        logger.error(f"Failed to save hunting clubs: {e}")
        return False


def set_hunting_club(club: str) -> bool:
    """Store a single hunting club (legacy, wraps set_hunting_clubs)."""
    return set_hunting_clubs([club])


def is_admin() -> bool:
    """Check if current user is admin (can see all clubs)."""
    config = get_config()
    return config.get("is_admin", False)


def set_admin(is_admin: bool) -> bool:
    """Set admin status."""
    return set_config_value("is_admin", is_admin)


def get_available_clubs() -> List[str]:
    """Get list of available hunting clubs."""
    config = get_config()
    custom_clubs = config.get("custom_clubs", [])
    return sorted(set(DEFAULT_CLUBS + custom_clubs))


def add_club(club: str) -> bool:
    """Add a new hunting club to the list."""
    config = get_config()
    custom_clubs = config.get("custom_clubs", [])
    if club not in custom_clubs and club not in DEFAULT_CLUBS:
        custom_clubs.append(club)
        return set_config_value("custom_clubs", custom_clubs)
    return True


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
