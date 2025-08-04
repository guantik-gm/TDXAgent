"""
Data formatting and normalization utilities for TDXAgent.

This module provides comprehensive data formatting, normalization,
and transformation utilities for consistent data handling across platforms.
"""

import re
import html
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
import logging

from utils.logger import TDXLogger
from utils.helpers import (
    format_timestamp, extract_urls, extract_mentions, 
    extract_hashtags, clean_text, generate_message_id
)
from utils.validators import validate_timestamp


class DataFormatter:
    """
    Comprehensive data formatting and normalization system.
    
    Features:
    - Platform-specific data normalization
    - Text cleaning and standardization
    - URL and media extraction
    - Timestamp normalization
    - Content sanitization
    """
    
    def __init__(self):
        self.logger = TDXLogger.get_logger("tdxagent.storage.formatter")
        
        # Platform-specific configurations
        self.platform_configs = {
            'twitter': {
                'max_text_length': 280,
                'mention_pattern': r'@([a-zA-Z0-9_]+)',
                'hashtag_pattern': r'#([a-zA-Z0-9_]+)',
                'url_pattern': r'https://t\.co/\w+'
            },
            'telegram': {
                'max_text_length': 4096,
                'mention_pattern': r'@([a-zA-Z0-9_]+)',
                'hashtag_pattern': r'#([a-zA-Z0-9_]+)',
                'url_pattern': r'https?://[^\s]+'
            },
            'discord': {
                'max_text_length': 2000,
                'mention_pattern': r'<@!?(\d+)>',
                'hashtag_pattern': r'#([a-zA-Z0-9_-]+)',
                'url_pattern': r'https?://[^\s]+'
            }
        }
    
    def normalize_text(self, text: str, platform: str = "") -> str:
        """
        Normalize text content for consistent processing.
        
        Args:
            text: Raw text content
            platform: Platform name for platform-specific normalization
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Decode HTML entities
        normalized = html.unescape(text)
        
        # Normalize Unicode
        normalized = normalized.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        
        # Platform-specific normalization
        if platform.lower() == 'twitter':
            # Handle Twitter-specific formatting
            normalized = self._normalize_twitter_text(normalized)
        elif platform.lower() == 'telegram':
            # Handle Telegram-specific formatting
            normalized = self._normalize_telegram_text(normalized)
        elif platform.lower() == 'discord':
            # Handle Discord-specific formatting
            normalized = self._normalize_discord_text(normalized)
        
        return normalized
    
    def _normalize_twitter_text(self, text: str) -> str:
        """Normalize Twitter-specific text formatting."""
        # Handle Twitter's t.co URLs
        text = re.sub(r'https://t\.co/\w+', lambda m: f'[TWITTER_URL:{m.group()}]', text)
        
        # Normalize Twitter mentions
        text = re.sub(r'@([a-zA-Z0-9_]+)', r'@\1', text)
        
        return text
    
    def _normalize_telegram_text(self, text: str) -> str:
        """Normalize Telegram-specific text formatting."""
        # Handle Telegram formatting (bold, italic, etc.)
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)  # Bold
        text = re.sub(r'__(.*?)__', r'_\1_', text)        # Italic
        text = re.sub(r'`(.*?)`', r'`\1`', text)          # Code
        
        return text
    
    def _normalize_discord_text(self, text: str) -> str:
        """Normalize Discord-specific text formatting."""
        # Handle Discord mentions
        text = re.sub(r'<@!?(\d+)>', r'@USER_\1', text)
        
        # Handle Discord channels
        text = re.sub(r'<#(\d+)>', r'#CHANNEL_\1', text)
        
        # Handle Discord roles
        text = re.sub(r'<@&(\d+)>', r'@ROLE_\1', text)
        
        # Handle Discord emojis
        text = re.sub(r'<:(.*?):\d+>', r':\1:', text)
        
        return text
    
    def normalize_timestamp(self, timestamp: Union[str, datetime, int, float], 
                           platform: str = "") -> str:
        """
        Normalize timestamp to ISO format.
        
        Args:
            timestamp: Raw timestamp
            platform: Platform name for platform-specific handling
            
        Returns:
            ISO formatted timestamp string
        """
        try:
            if isinstance(timestamp, datetime):
                # Ensure timezone info
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return timestamp.isoformat()
            
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                return dt.isoformat()
            
            elif isinstance(timestamp, str):
                # Parse string timestamp
                if not timestamp:
                    return datetime.now(timezone.utc).isoformat()
                
                # Handle common formats
                if timestamp.endswith('Z'):
                    timestamp = timestamp[:-1] + '+00:00'
                
                # Platform-specific timestamp handling
                if platform.lower() == 'twitter':
                    # Twitter uses specific format
                    timestamp = self._normalize_twitter_timestamp(timestamp)
                elif platform.lower() == 'telegram':
                    # Telegram uses Unix timestamps
                    timestamp = self._normalize_telegram_timestamp(timestamp)
                
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            
            else:
                # Fallback to current time
                return datetime.now(timezone.utc).isoformat()
                
        except Exception as e:
            self.logger.warning(f"Failed to normalize timestamp {timestamp}: {e}")
            return datetime.now(timezone.utc).isoformat()
    
    def _normalize_twitter_timestamp(self, timestamp: str) -> str:
        """Normalize Twitter timestamp format."""
        # Twitter format: "Wed Oct 05 20:12:34 +0000 2022"
        try:
            dt = datetime.strptime(timestamp, "%a %b %d %H:%M:%S %z %Y")
            return dt.isoformat()
        except ValueError:
            return timestamp
    
    def _normalize_telegram_timestamp(self, timestamp: str) -> str:
        """Normalize Telegram timestamp format."""
        # Telegram often uses Unix timestamps as strings
        try:
            if timestamp.isdigit():
                dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
                return dt.isoformat()
        except ValueError:
            pass
        return timestamp
    
    def extract_media_urls(self, content: Dict[str, Any], platform: str = "") -> List[str]:
        """
        Extract media URLs from content.
        
        Args:
            content: Content dictionary
            platform: Platform name
            
        Returns:
            List of media URLs
        """
        media_urls = []
        
        # Extract from text content
        text = content.get('text', '')
        if text:
            urls = extract_urls(text)
            media_urls.extend(self._filter_media_urls(urls, platform))
        
        # Extract from media field
        media_list = content.get('media', [])
        if isinstance(media_list, list):
            for media_item in media_list:
                if isinstance(media_item, str):
                    media_urls.append(media_item)
                elif isinstance(media_item, dict):
                    url = media_item.get('url') or media_item.get('media_url')
                    if url:
                        media_urls.append(url)
        
        # Extract from attachments (Discord)
        attachments = content.get('attachments', [])
        if isinstance(attachments, list):
            for attachment in attachments:
                if isinstance(attachment, dict):
                    url = attachment.get('url') or attachment.get('proxy_url')
                    if url:
                        media_urls.append(url)
        
        return list(set(media_urls))  # Remove duplicates
    
    def _filter_media_urls(self, urls: List[str], platform: str) -> List[str]:
        """Filter URLs to only include media URLs."""
        media_urls = []
        
        media_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
                           '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
                           '.mp3', '.wav', '.flac', '.aac', '.ogg'}
        
        for url in urls:
            # Check file extension
            parsed = urlparse(url)
            path = unquote(parsed.path).lower()
            
            if any(path.endswith(ext) for ext in media_extensions):
                media_urls.append(url)
                continue
            
            # Platform-specific media URL patterns
            if platform.lower() == 'twitter':
                if 'pbs.twimg.com' in url or 'video.twimg.com' in url:
                    media_urls.append(url)
            elif platform.lower() == 'telegram':
                if 'cdn.telegram.org' in url or 'telegram.org/file' in url:
                    media_urls.append(url)
            elif platform.lower() == 'discord':
                if 'cdn.discordapp.com' in url or 'media.discordapp.net' in url:
                    media_urls.append(url)
        
        return media_urls
    
    def format_message_content(self, raw_content: Dict[str, Any], 
                              platform: str) -> Dict[str, Any]:
        """
        Format and normalize message content.
        
        Args:
            raw_content: Raw content data
            platform: Platform name
            
        Returns:
            Formatted content dictionary
        """
        # Extract and normalize text
        text = raw_content.get('text', '') or raw_content.get('content', '') or ''
        normalized_text = self.normalize_text(text, platform)
        
        # Extract HTML content if available
        html_content = raw_content.get('html', '') or raw_content.get('html_content', '')
        if html_content:
            html_content = self.normalize_text(html_content, platform)
        
        # Extract media URLs
        media_urls = self.extract_media_urls(raw_content, platform)
        
        # Extract mentions and hashtags
        mentions = extract_mentions(normalized_text, platform)
        hashtags = extract_hashtags(normalized_text)
        urls = extract_urls(normalized_text)
        
        return {
            'text': normalized_text,
            'html': html_content,
            'media': media_urls,
            'mentions': mentions,
            'hashtags': hashtags,
            'urls': urls,
            'length': len(normalized_text)
        }
    
    def format_author_info(self, raw_author: Dict[str, Any], 
                          platform: str) -> Dict[str, Any]:
        """
        Format and normalize author information.
        
        Args:
            raw_author: Raw author data
            platform: Platform name
            
        Returns:
            Formatted author dictionary
        """
        # Extract basic info
        name = raw_author.get('name', '') or raw_author.get('username', '') or 'Unknown'
        user_id = str(raw_author.get('id', '') or raw_author.get('user_id', ''))
        
        # Normalize name
        name = self.normalize_text(name, platform)
        
        # Extract additional info
        display_name = raw_author.get('display_name', '') or raw_author.get('full_name', '')
        if display_name:
            display_name = self.normalize_text(display_name, platform)
        
        avatar_url = raw_author.get('avatar_url', '') or raw_author.get('profile_image_url', '')
        
        return {
            'name': name,
            'id': user_id,
            'display_name': display_name,
            'avatar_url': avatar_url,
            'platform_specific': {
                key: value for key, value in raw_author.items()
                if key not in ['name', 'id', 'display_name', 'avatar_url']
            }
        }
    
    def format_metadata(self, raw_metadata: Dict[str, Any], 
                       platform: str) -> Dict[str, Any]:
        """
        Format and normalize metadata.
        
        Args:
            raw_metadata: Raw metadata
            platform: Platform name
            
        Returns:
            Formatted metadata dictionary
        """
        # Normalize timestamp
        posted_at = raw_metadata.get('posted_at') or raw_metadata.get('timestamp') or raw_metadata.get('created_at')
        normalized_timestamp = self.normalize_timestamp(posted_at, platform)
        
        # Extract URL
        message_url = raw_metadata.get('url', '') or raw_metadata.get('message_url', '')
        
        # Extract reply information
        reply_to = raw_metadata.get('reply_to', '') or raw_metadata.get('in_reply_to', '')
        
        # Extract reactions/engagement
        reactions = raw_metadata.get('reactions', {})
        if not isinstance(reactions, dict):
            reactions = {}
        
        # Platform-specific metadata
        platform_metadata = {}
        if platform.lower() == 'twitter':
            platform_metadata = {
                'retweet_count': raw_metadata.get('retweet_count', 0),
                'like_count': raw_metadata.get('like_count', 0),
                'reply_count': raw_metadata.get('reply_count', 0),
                'quote_count': raw_metadata.get('quote_count', 0)
            }
        elif platform.lower() == 'telegram':
            platform_metadata = {
                'message_id': raw_metadata.get('message_id', ''),
                'chat_id': raw_metadata.get('chat_id', ''),
                'forward_from': raw_metadata.get('forward_from', '')
            }
        elif platform.lower() == 'discord':
            platform_metadata = {
                'message_id': raw_metadata.get('message_id', ''),
                'channel_id': raw_metadata.get('channel_id', ''),
                'guild_id': raw_metadata.get('guild_id', ''),
                'thread_id': raw_metadata.get('thread_id', '')
            }
        
        return {
            'posted_at': normalized_timestamp,
            'message_url': message_url,
            'reply_to': reply_to,
            'reactions': reactions,
            'scraped_at': datetime.now(timezone.utc).isoformat(),
            'platform_specific': platform_metadata
        }
    
    def format_context(self, raw_context: Dict[str, Any], 
                      platform: str) -> Dict[str, Any]:
        """
        Format and normalize context information.
        
        Args:
            raw_context: Raw context data
            platform: Platform name
            
        Returns:
            Formatted context dictionary
        """
        # Extract common context fields
        channel = raw_context.get('channel', '') or raw_context.get('chat_title', '')
        server = raw_context.get('server', '') or raw_context.get('guild_name', '')
        thread = raw_context.get('thread', '') or raw_context.get('thread_name', '')
        group = raw_context.get('group', '') or raw_context.get('group_name', '')
        
        # Normalize text fields
        if channel:
            channel = self.normalize_text(channel, platform)
        if server:
            server = self.normalize_text(server, platform)
        if thread:
            thread = self.normalize_text(thread, platform)
        if group:
            group = self.normalize_text(group, platform)
        
        return {
            'channel': channel,
            'server': server,
            'thread': thread,
            'group': group,
            'platform_specific': {
                key: value for key, value in raw_context.items()
                if key not in ['channel', 'server', 'thread', 'group']
            }
        }
