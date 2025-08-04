"""
Configuration validation utilities for TDXAgent.

This module provides comprehensive validation for configuration files,
ensuring that all required settings are present and properly formatted.
"""

import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from .logger import TDXLogger


class ValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigValidator:
    """
    Configuration validator for TDXAgent.
    
    Validates configuration files against required schemas and
    provides detailed error reporting.
    """
    
    def __init__(self):
        self.logger = TDXLogger.get_logger("tdxagent.config_validator")
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate complete configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Validate main sections
        self._validate_settings(config.get('settings', {}))
        self._validate_platforms(config.get('platforms', {}))
        self._validate_llm_config(config.get('llm', {}))
        self._validate_prompts(config.get('prompts', {}))
        self._validate_output_config(config.get('output', {}))
        
        # Report results
        if self.warnings:
            for warning in self.warnings:
                self.logger.warning(f"Config warning: {warning}")
        
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(self.errors)
            raise ValidationError(error_msg)
        
        self.logger.info("Configuration validation passed")
        return True
    
    def _validate_settings(self, settings: Dict[str, Any]):
        """Validate settings section."""
        required_fields = ['default_hours_to_fetch', 'data_directory', 'log_level']
        
        for field in required_fields:
            if field not in settings:
                self.errors.append(f"Missing required settings field: {field}")
        
        # Validate specific fields
        if 'default_hours_to_fetch' in settings:
            hours = settings['default_hours_to_fetch']
            if not isinstance(hours, int) or hours < 1 or hours > 168:
                self.errors.append(
                    f"default_hours_to_fetch must be an integer between 1 and 168, got: {hours}"
                )
        
        if 'log_level' in settings:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if settings['log_level'] not in valid_levels:
                self.errors.append(
                    f"log_level must be one of {valid_levels}, got: {settings['log_level']}"
                )
        
        if 'data_directory' in settings:
            data_dir = Path(settings['data_directory'])
            if not data_dir.parent.exists():
                self.warnings.append(
                    f"Parent directory of data_directory does not exist: {data_dir.parent}"
                )
    
    def _validate_platforms(self, platforms: Dict[str, Any]):
        """Validate platforms section."""
        supported_platforms = ['twitter', 'telegram', 'discord']
        
        for platform_name, platform_config in platforms.items():
            if platform_name not in supported_platforms:
                self.warnings.append(f"Unknown platform: {platform_name}")
                continue
            
            if not isinstance(platform_config, dict):
                self.errors.append(f"Platform {platform_name} config must be a dictionary")
                continue
            
            # Validate platform-specific configurations
            if platform_name == 'twitter':
                self._validate_twitter_config(platform_config)
            elif platform_name == 'telegram':
                self._validate_telegram_config(platform_config)
            elif platform_name == 'discord':
                self._validate_discord_config(platform_config)
    
    def _validate_twitter_config(self, config: Dict[str, Any]):
        """Validate Twitter configuration."""
        if config.get('enabled', False):
            if 'delay_range' in config:
                delay_range = config['delay_range']
                if not isinstance(delay_range, list) or len(delay_range) != 2:
                    self.errors.append("Twitter delay_range must be a list of 2 numbers")
                elif delay_range[0] >= delay_range[1]:
                    self.errors.append("Twitter delay_range first value must be less than second")
            
            if 'max_scrolls' in config:
                max_scrolls = config['max_scrolls']
                if not isinstance(max_scrolls, int) or max_scrolls < 1:
                    self.errors.append("Twitter max_scrolls must be a positive integer")
    
    def _validate_telegram_config(self, config: Dict[str, Any]):
        """Validate Telegram configuration."""
        if config.get('enabled', False):
            required_fields = ['api_id', 'api_hash']
            
            for field in required_fields:
                if field not in config or not config[field]:
                    self.errors.append(f"Telegram {field} is required when enabled")
            
            if 'api_id' in config:
                try:
                    int(config['api_id'])
                except (ValueError, TypeError):
                    self.errors.append("Telegram api_id must be a valid integer")
            
            if 'group_blacklist' in config:
                blacklist = config['group_blacklist']
                if not isinstance(blacklist, list):
                    self.errors.append("Telegram group_blacklist must be a list")
    
    def _validate_discord_config(self, config: Dict[str, Any]):
        """Validate Discord configuration."""
        if config.get('enabled', False):
            valid_modes = ['safe', 'experimental']
            mode = config.get('mode', 'safe')
            
            if mode not in valid_modes:
                self.errors.append(f"Discord mode must be one of {valid_modes}, got: {mode}")
            
            if mode == 'experimental':
                self.warnings.append(
                    "Discord experimental mode is risky and may result in account suspension"
                )
            
            if 'export_path' in config:
                export_path = Path(config['export_path'])
                if not export_path.exists():
                    self.warnings.append(
                        f"Discord export_path does not exist: {export_path}"
                    )
    
    def _validate_llm_config(self, llm_config: Dict[str, Any]):
        """Validate LLM configuration."""
        if not llm_config:
            self.errors.append("LLM configuration is required")
            return
        
        valid_providers = ['openai', 'gemini']
        provider = llm_config.get('provider')
        
        if not provider:
            self.errors.append("LLM provider is required")
        elif provider not in valid_providers:
            self.errors.append(f"LLM provider must be one of {valid_providers}, got: {provider}")
        
        # Validate provider-specific configs
        if provider == 'openai':
            self._validate_openai_config(llm_config.get('openai', {}))
        elif provider == 'gemini':
            self._validate_gemini_config(llm_config.get('gemini', {}))
        
        # Validate common LLM settings
        if 'batch_size' in llm_config:
            batch_size = llm_config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1:
                self.errors.append("LLM batch_size must be a positive integer")
        
        if 'max_tokens' in llm_config:
            max_tokens = llm_config['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens < 1:
                self.errors.append("LLM max_tokens must be a positive integer")
    
    def _validate_openai_config(self, config: Dict[str, Any]):
        """Validate OpenAI configuration."""
        if 'api_key' not in config or not config['api_key']:
            self.errors.append("OpenAI api_key is required")
        elif config['api_key'] in ['sk-your-openai-api-key', 'sk-...']:
            self.errors.append("OpenAI api_key appears to be a placeholder")
        
        if 'base_url' in config:
            base_url = config['base_url']
            if not self._is_valid_url(base_url):
                self.errors.append(f"OpenAI base_url is not a valid URL: {base_url}")
        
        if 'model' in config:
            model = config['model']
            if not isinstance(model, str) or not model:
                self.errors.append("OpenAI model must be a non-empty string")
    
    def _validate_gemini_config(self, config: Dict[str, Any]):
        """Validate Gemini configuration."""
        if 'api_key' not in config or not config['api_key']:
            self.errors.append("Gemini api_key is required")
        elif config['api_key'] in ['your-gemini-api-key', '...']:
            self.errors.append("Gemini api_key appears to be a placeholder")
        
        if 'model' in config:
            model = config['model']
            if not isinstance(model, str) or not model:
                self.errors.append("Gemini model must be a non-empty string")
    
    def _validate_prompts(self, prompts: Dict[str, Any]):
        """Validate prompts section."""
        if not prompts:
            self.warnings.append("No prompts configured, using defaults")
            return
        
        for platform, prompt_template in prompts.items():
            if not isinstance(prompt_template, str):
                self.errors.append(f"Prompt for {platform} must be a string")
            elif '{data}' not in prompt_template:
                self.warnings.append(
                    f"Prompt for {platform} does not contain {{data}} placeholder"
                )
    
    def _validate_output_config(self, output_config: Dict[str, Any]):
        """Validate output configuration."""
        if 'format' in output_config:
            valid_formats = ['markdown', 'json', 'text']
            format_type = output_config['format']
            if format_type not in valid_formats:
                self.errors.append(
                    f"Output format must be one of {valid_formats}, got: {format_type}"
                )
        
        if 'filename_template' in output_config:
            template = output_config['filename_template']
            if not isinstance(template, str):
                self.errors.append("Output filename_template must be a string")
            elif not any(placeholder in template for placeholder in ['{platform}', '{timestamp}']):
                self.warnings.append(
                    "Output filename_template should contain {platform} and/or {timestamp} placeholders"
                )
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'is_valid': len(self.errors) == 0
        }