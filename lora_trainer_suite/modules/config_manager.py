"""
Configuration Manager
Manages application settings and configuration persistence
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """Manage application configuration"""

    DEFAULT_CONFIG = {
        'device': 'cuda',
        'vram_optimization': 'none',
        'cpu_threads': 12,
        'cache_dir': './models',
        'dataset_path': '',
        'output_dir': './output',
        'default_resolution': 512,
        'default_batch_size': 1,
        'default_learning_rate': 1e-4,
        'default_lora_rank': 16,
        'default_lora_alpha': 32,
        'clip_model': 'ViT-L-14/openai',
        'qwen_model': 'Qwen/Qwen2-VL-7B-Instruct',
        'use_flash_attention': True,
        'use_4bit_quantization': True,
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Configuration Manager

        Args:
            config_path: Path to config file (default: ~/.lora_trainer/config.json)
        """
        if config_path is None:
            config_dir = Path.home() / '.lora_trainer'
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / 'config.json'

        self.config_path = Path(config_path)
        self.config = self.DEFAULT_CONFIG.copy()

        # Load existing config
        self.load()

    def load(self):
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)

                # Merge with defaults (in case new keys were added)
                self.config.update(loaded_config)

                print(f"✓ Configuration loaded from {self.config_path}")

            except Exception as e:
                print(f"⚠ Failed to load config: {e}")
                print("  Using default configuration")

    def save(self):
        """Save configuration to file"""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            print(f"✓ Configuration saved to {self.config_path}")

        except Exception as e:
            print(f"✗ Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set configuration value

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values

        Args:
            updates: Dictionary of updates
        """
        self.config.update(updates)

    def reset(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
        print("✓ Configuration reset to defaults")

    def export(self, export_path: str):
        """
        Export configuration to file

        Args:
            export_path: Path to export file
        """
        export_path = Path(export_path)

        with open(export_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✓ Configuration exported to {export_path}")

    def import_config(self, import_path: str):
        """
        Import configuration from file

        Args:
            import_path: Path to import file
        """
        import_path = Path(import_path)

        if not import_path.exists():
            raise FileNotFoundError(f"Config file not found: {import_path}")

        with open(import_path, 'r') as f:
            imported_config = json.load(f)

        self.config.update(imported_config)
        self.save()

        print(f"✓ Configuration imported from {import_path}")

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.config.copy()

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting"""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config"""
        return key in self.config


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configuration Manager CLI")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--set", nargs=2, metavar=('KEY', 'VALUE'), help="Set configuration value")
    parser.add_argument("--reset", action="store_true", help="Reset to defaults")
    parser.add_argument("--export", help="Export configuration to file")
    parser.add_argument("--import", dest='import_file', help="Import configuration from file")

    args = parser.parse_args()

    config = ConfigManager()

    if args.show:
        print("\n=== Current Configuration ===")
        print(json.dumps(config.get_all(), indent=2))

    if args.set:
        key, value = args.set
        # Try to parse value as JSON for proper typing
        try:
            value = json.loads(value)
        except:
            pass  # Keep as string

        config.set(key, value)
        config.save()
        print(f"✓ Set {key} = {value}")

    if args.reset:
        config.reset()

    if args.export:
        config.export(args.export)

    if args.import_file:
        config.import_config(args.import_file)
