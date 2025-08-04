"""
Helper utilities for TDXAgent.

This module provides common utility functions used throughout the application.
"""

import re
import hashlib
import asyncio
import aiofiles
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import json
import unicodedata
import logging


logger = logging.getLogger(__name__)


def format_timestamp(timestamp: Optional[Union[datetime, str, int, float]] = None, 
                    format_type: str = "iso", timezone_offset: int = 8) -> str:
    """
    Format a timestamp into a standardized string format.
    
    Args:
        timestamp: Timestamp to format (defaults to current time)
        format_type: Format type ('iso', 'filename', 'human')
        timezone_offset: Timezone offset from UTC in hours (default: 8 for UTC+8)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    elif isinstance(timestamp, str):
        # Try to parse ISO format
        try:
            if timestamp.endswith('Z'):
                timestamp = timestamp[:-1] + '+00:00'
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Failed to parse timestamp: {timestamp}")
            dt = datetime.now(timezone.utc)
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    
    # Convert to specified timezone (default UTC+8)
    from datetime import timedelta
    target_tz = timezone(timedelta(hours=timezone_offset))
    dt = dt.astimezone(target_tz)
    
    if format_type == "iso":
        return dt.isoformat()
    elif format_type == "filename":
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    elif format_type == "human":
        tz_str = f"UTC+{timezone_offset}" if timezone_offset >= 0 else f"UTC{timezone_offset}"
        return dt.strftime(f"%Y-%m-%d %H:%M:%S {tz_str}")
    else:
        return dt.isoformat()


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename for safe filesystem use.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if unicodedata.category(char)[0] != 'C')
    
    # Normalize unicode
    sanitized = unicodedata.normalize('NFKD', sanitized)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = max_length - len(ext)
        if max_name_length > 0:
            sanitized = name[:max_name_length] + ext
        else:
            sanitized = sanitized[:max_length]
    
    return sanitized


def generate_message_id(platform: str, original_id: str, author_id: str = "") -> str:
    """
    Generate a unique message ID for TDXAgent.
    
    Args:
        platform: Platform name
        original_id: Original message ID from platform
        author_id: Author ID (optional)
        
    Returns:
        Unique message ID
    """
    # Create a hash of the platform, original ID, and author ID
    content = f"{platform}:{original_id}:{author_id}"
    hash_object = hashlib.md5(content.encode())
    return f"tdx_{platform}_{hash_object.hexdigest()[:12]}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to search for URLs
        
    Returns:
        List of found URLs
    """
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)


def extract_mentions(text: str, platform: str = "twitter") -> List[str]:
    """
    Extract mentions from text based on platform conventions.
    
    Args:
        text: Text to search for mentions
        platform: Platform type (affects mention format)
        
    Returns:
        List of mentioned usernames
    """
    if platform.lower() in ["twitter", "x"]:
        pattern = r'@([a-zA-Z0-9_]+)'
    elif platform.lower() == "telegram":
        pattern = r'@([a-zA-Z0-9_]+)'
    elif platform.lower() == "discord":
        pattern = r'<@!?(\d+)>'
    else:
        pattern = r'@([a-zA-Z0-9_]+)'
    
    return re.findall(pattern, text)


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text.
    
    Args:
        text: Text to search for hashtags
        
    Returns:
        List of hashtags (without # symbol)
    """
    pattern = r'#([a-zA-Z0-9_]+)'
    return re.findall(pattern, text)


def clean_text(text: str, remove_urls: bool = False, remove_mentions: bool = False) -> str:
    """
    Clean text by removing or normalizing various elements.
    
    Args:
        text: Text to clean
        remove_urls: Whether to remove URLs
        remove_mentions: Whether to remove mentions
        
    Returns:
        Cleaned text
    """
    cleaned = text
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove URLs if requested
    if remove_urls:
        cleaned = re.sub(r'http[s]?://\S+', '', cleaned)
    
    # Remove mentions if requested
    if remove_mentions:
        cleaned = re.sub(r'@\w+', '', cleaned)
    
    # Remove extra whitespace again
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


async def ensure_directory_async(path: Union[str, Path]) -> Path:
    """
    Asynchronously ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


async def read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Asynchronously read a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)


async def write_json_file(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
    """
    Asynchronously write data to a JSON file.
    
    Args:
        file_path: Path to JSON file
        data: Data to write
    """
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        content = json.dumps(data, ensure_ascii=False, indent=2)
        await f.write(content)


async def append_jsonl_file(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
    """
    Asynchronously append data to a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        data: Data to append
    """
    async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
        line = json.dumps(data, ensure_ascii=False) + '\n'
        await f.write(line)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity based on common words.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


async def rate_limited_gather(*coroutines, rate_limit: float = 1.0):
    """
    Execute coroutines with rate limiting.
    
    Args:
        *coroutines: Coroutines to execute
        rate_limit: Minimum delay between executions
        
    Returns:
        List of results
    """
    results = []
    
    for i, coro in enumerate(coroutines):
        if i > 0:
            await asyncio.sleep(rate_limit)
        
        result = await coro
        results.append(result)
    
    return results


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename
        
    Returns:
        File extension (including dot)
    """
    return Path(filename).suffix.lower()


def is_media_file(filename: str) -> bool:
    """
    Check if a file is a media file based on extension.
    
    Args:
        filename: Filename to check
        
    Returns:
        True if it's a media file
    """
    media_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',  # Images
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',   # Videos
        '.mp3', '.wav', '.flac', '.aac', '.ogg'            # Audio
    }
    
    return get_file_extension(filename) in media_extensions
