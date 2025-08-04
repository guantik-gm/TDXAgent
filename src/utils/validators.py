"""
Data validation utilities for TDXAgent.

This module provides validation functions for ensuring data integrity
and format compliance across the application.
"""

import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


def validate_message(message: Dict[str, Any]) -> bool:
    """
    Validate a message against the standard TDXAgent format.
    
    Args:
        message: Message dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required top-level fields
        required_fields = ['id', 'platform', 'author', 'content', 'metadata', 'context']
        for field in required_fields:
            if field not in message:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate author structure
        author = message.get('author', {})
        if not isinstance(author, dict):
            logger.warning("Author field must be a dictionary")
            return False
        
        author_required = ['name', 'id']
        for field in author_required:
            if field not in author:
                logger.warning(f"Missing required author field: {field}")
                return False
        
        # Validate content structure
        content = message.get('content', {})
        if not isinstance(content, dict):
            logger.warning("Content field must be a dictionary")
            return False
        
        if 'text' not in content:
            logger.warning("Missing required content field: text")
            return False
        
        # Validate metadata structure
        metadata = message.get('metadata', {})
        if not isinstance(metadata, dict):
            logger.warning("Metadata field must be a dictionary")
            return False
        
        if 'posted_at' not in metadata:
            logger.warning("Missing required metadata field: posted_at")
            return False
        
        # Validate timestamp format
        posted_at = metadata.get('posted_at')
        if not validate_timestamp(posted_at):
            logger.warning(f"Invalid timestamp format: {posted_at}")
            return False
        
        # Validate context structure
        context = message.get('context', {})
        if not isinstance(context, dict):
            logger.warning("Context field must be a dictionary")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating message: {e}")
        return False


def validate_timestamp(timestamp: Union[str, datetime, int, float]) -> bool:
    """
    Validate timestamp format.
    
    Args:
        timestamp: Timestamp to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if isinstance(timestamp, datetime):
            return True
        elif isinstance(timestamp, (int, float)):
            # Unix timestamp
            datetime.fromtimestamp(timestamp)
            return True
        elif isinstance(timestamp, str):
            # ISO format string
            if timestamp.endswith('Z'):
                timestamp = timestamp[:-1] + '+00:00'
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        else:
            return False
    except (ValueError, TypeError, OSError):
        return False


def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate application configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Validate settings section
        settings = config.get('settings', {})
        if not isinstance(settings, dict):
            errors.append("Settings section must be a dictionary")
        else:
            # Check numeric settings
            numeric_settings = {
                'default_hours_to_fetch': (1, 168),  # 1 hour to 1 week
                'max_retries': (0, 10),
                'max_concurrent_tasks': (1, 20)
            }
            
            for setting, (min_val, max_val) in numeric_settings.items():
                value = settings.get(setting)
                if value is not None:
                    if not isinstance(value, int) or not (min_val <= value <= max_val):
                        errors.append(
                            f"Setting '{setting}' must be an integer between {min_val} and {max_val}"
                        )
        
        # Validate platforms section
        platforms = config.get('platforms', {})
        if not isinstance(platforms, dict):
            errors.append("Platforms section must be a dictionary")
        else:
            # Validate individual platform configs
            for platform_name, platform_config in platforms.items():
                if not isinstance(platform_config, dict):
                    errors.append(f"Platform '{platform_name}' config must be a dictionary")
                    continue
                
                # Platform-specific validation
                if platform_name == 'telegram':
                    telegram_errors = validate_telegram_config(platform_config)
                    errors.extend(telegram_errors)
                elif platform_name == 'discord':
                    discord_errors = validate_discord_config(platform_config)
                    errors.extend(discord_errors)
                elif platform_name == 'twitter':
                    twitter_errors = validate_twitter_config(platform_config)
                    errors.extend(twitter_errors)
        
        # Validate LLM section
        llm = config.get('llm', {})
        if not isinstance(llm, dict):
            errors.append("LLM section must be a dictionary")
        else:
            llm_errors = validate_llm_config(llm)
            errors.extend(llm_errors)
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Unexpected error during validation: {e}")
        return False, errors


def validate_telegram_config(config: Dict[str, Any]) -> List[str]:
    """Validate Telegram-specific configuration."""
    errors = []
    
    if config.get('enabled', False):
        api_id = config.get('api_id')
        api_hash = config.get('api_hash')
        
        if not api_id or not isinstance(api_id, (str, int)):
            errors.append("Telegram API ID is required and must be a string or integer")
        
        if not api_hash or not isinstance(api_hash, str):
            errors.append("Telegram API hash is required and must be a string")
        
        # Validate API ID format (should be numeric)
        if api_id and isinstance(api_id, str) and not api_id.isdigit():
            errors.append("Telegram API ID should be numeric")
        
        # Validate API hash format (should be 32 characters)
        if api_hash and len(api_hash) != 32:
            errors.append("Telegram API hash should be 32 characters long")
    
    return errors


def validate_discord_config(config: Dict[str, Any]) -> List[str]:
    """Validate Discord-specific configuration."""
    errors = []
    
    if config.get('enabled', False):
        mode = config.get('mode', 'safe')
        if mode not in ['safe', 'experimental']:
            errors.append("Discord mode must be either 'safe' or 'experimental'")
        
        if mode == 'experimental':
            experimental = config.get('experimental', {})
            token = experimental.get('token')
            if not token or not isinstance(token, str):
                errors.append("Discord token is required for experimental mode")
    
    return errors


def validate_twitter_config(config: Dict[str, Any]) -> List[str]:
    """Validate Twitter-specific configuration."""
    errors = []
    
    if config.get('enabled', False):
        delay_range = config.get('delay_range', [2, 5])
        if not isinstance(delay_range, list) or len(delay_range) != 2:
            errors.append("Twitter delay_range must be a list of two numbers")
        elif not all(isinstance(x, (int, float)) for x in delay_range):
            errors.append("Twitter delay_range values must be numbers")
        elif delay_range[0] >= delay_range[1]:
            errors.append("Twitter delay_range first value must be less than second")
        
        max_scrolls = config.get('max_scrolls', 10)
        if not isinstance(max_scrolls, int) or max_scrolls < 1:
            errors.append("Twitter max_scrolls must be a positive integer")
    
    return errors


def validate_llm_config(config: Dict[str, Any]) -> List[str]:
    """Validate LLM configuration."""
    errors = []
    
    provider = config.get('provider', 'openai')
    if provider not in ['openai', 'gemini']:
        errors.append("LLM provider must be either 'openai' or 'gemini'")
    
    batch_size = config.get('batch_size', 50)
    if not isinstance(batch_size, int) or batch_size < 1:
        errors.append("LLM batch_size must be a positive integer")
    
    max_tokens = config.get('max_tokens', 4000)
    if not isinstance(max_tokens, int) or max_tokens < 100:
        errors.append("LLM max_tokens must be an integer >= 100")
    
    # Validate provider-specific configs
    if provider == 'openai':
        openai_config = config.get('openai', {})
        api_key = openai_config.get('api_key', '')
        if not api_key or not isinstance(api_key, str):
            errors.append("OpenAI API key is required")
        elif not api_key.startswith('sk-'):
            errors.append("OpenAI API key should start with 'sk-'")
    
    elif provider == 'gemini':
        gemini_config = config.get('gemini', {})
        api_key = gemini_config.get('api_key', '')
        if not api_key or not isinstance(api_key, str):
            errors.append("Gemini API key is required")
    
    return errors


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def validate_file_path(path: str, must_exist: bool = False) -> bool:
    """
    Validate file path format and optionally check existence.
    
    Args:
        path: File path to validate
        must_exist: Whether the file must exist
        
    Returns:
        True if valid, False otherwise
    """
    try:
        from pathlib import Path
        path_obj = Path(path)
        
        # Check for invalid characters (basic check)
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in str(path_obj) for char in invalid_chars):
            return False
        
        if must_exist:
            return path_obj.exists()
        
        return True
        
    except Exception:
        return False


def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize a string for safe use in filenames or logs.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length (optional)
        
    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    sanitized = re.sub(r'\s+', ' ', sanitized)  # Normalize whitespace
    sanitized = sanitized.strip()
    
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length-3] + '...'
    
    return sanitized


def validate_message_batch(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate a batch of messages and return valid ones with error list.
    
    Args:
        messages: List of messages to validate
        
    Returns:
        Tuple of (valid_messages, error_messages)
    """
    valid_messages = []
    errors = []
    
    for i, message in enumerate(messages):
        if validate_message(message):
            valid_messages.append(message)
        else:
            errors.append(f"Message {i} failed validation")
    
    return valid_messages, errors
