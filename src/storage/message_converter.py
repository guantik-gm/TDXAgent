"""
Unified message format converter for TDXAgent.

This module converts platform-specific message data into the standardized
TDXAgent message format for consistent processing and storage.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from utils.logger import TDXLogger
from utils.helpers import generate_message_id
from utils.validators import validate_message
from storage.data_formatter import DataFormatter


class MessageConverter:
    """
    Unified message format converter.
    
    Converts platform-specific message data into the standardized TDXAgent format:
    {
        "id": "unique_message_id",
        "platform": "twitter|telegram|discord",
        "author": {
            "name": "username",
            "id": "user_id",
            "display_name": "display_name",
            "avatar_url": "avatar_url"
        },
        "content": {
            "text": "message_text",
            "html": "html_content",
            "media": ["media_urls"],
            "mentions": ["mentioned_users"],
            "hashtags": ["hashtags"],
            "urls": ["urls"]
        },
        "metadata": {
            "posted_at": "iso_timestamp",
            "message_url": "original_url",
            "reply_to": "replied_message_id",
            "reactions": {"emoji": count},
            "scraped_at": "iso_timestamp"
        },
        "context": {
            "channel": "channel_name",
            "server": "server_name",
            "thread": "thread_name",
            "group": "group_name"
        }
    }
    """
    
    def __init__(self):
        self.logger = TDXLogger.get_logger("tdxagent.storage.converter")
        self.formatter = DataFormatter()
        
        # Platform-specific converters
        self.converters = {
            'twitter': self._convert_twitter_message,
            'telegram': self._convert_telegram_message,
            'discord': self._convert_discord_message
        }
    
    def convert_message(self, raw_message: Dict[str, Any], 
                       platform: str) -> Optional[Dict[str, Any]]:
        """
        Convert a platform-specific message to unified format.
        
        Args:
            raw_message: Raw message data from platform
            platform: Platform name
            
        Returns:
            Converted message in unified format or None if conversion fails
        """
        try:
            platform_lower = platform.lower()
            
            if platform_lower not in self.converters:
                self.logger.error(f"Unsupported platform: {platform}")
                return None
            
            # Use platform-specific converter
            converter = self.converters[platform_lower]
            converted_message = converter(raw_message)
            
            if not converted_message:
                self.logger.warning(f"Failed to convert {platform} message")
                return None
            
            # Validate the converted message
            if not validate_message(converted_message):
                self.logger.warning(f"Converted {platform} message failed validation")
                return None
            
            self.logger.debug(f"Successfully converted {platform} message: {converted_message.get('id')}")
            return converted_message
            
        except Exception as e:
            self.logger.error(f"Error converting {platform} message: {e}")
            return None
    
    def convert_messages_batch(self, raw_messages: List[Dict[str, Any]], 
                              platform: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Convert multiple messages in batch.
        
        Args:
            raw_messages: List of raw message data
            platform: Platform name
            
        Returns:
            Tuple of (converted_messages, error_messages)
        """
        converted_messages = []
        errors = []
        
        for i, raw_message in enumerate(raw_messages):
            try:
                converted = self.convert_message(raw_message, platform)
                if converted:
                    converted_messages.append(converted)
                else:
                    errors.append(f"Message {i}: Conversion failed")
            except Exception as e:
                errors.append(f"Message {i}: {str(e)}")
        
        self.logger.info(
            f"Batch conversion complete: {len(converted_messages)}/{len(raw_messages)} successful"
        )
        
        return converted_messages, errors
    
    def _convert_twitter_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Twitter/X message to unified format."""
        try:
            # Extract basic message info
            tweet_id = raw_message.get('id') or raw_message.get('tweet_id', '')
            if not tweet_id:
                self.logger.warning("Twitter message missing ID")
                return None
            
            # Generate unified ID
            unified_id = generate_message_id('twitter', str(tweet_id))
            
            # Extract author information
            author_data = raw_message.get('author', {}) or raw_message.get('user', {})
            if not author_data:
                self.logger.warning("Twitter message missing author data")
                return None
            
            formatted_author = self.formatter.format_author_info(author_data, 'twitter')
            
            # Extract content
            content_data = {
                'text': raw_message.get('text', '') or raw_message.get('full_text', ''),
                'html': raw_message.get('html', ''),
                'media': raw_message.get('media_urls', []) or [],
                'attachments': raw_message.get('attachments', [])
            }
            
            # Handle Twitter entities (mentions, hashtags, URLs)
            entities = raw_message.get('entities', {})
            if entities:
                # Extract mentions from entities
                mentions = entities.get('user_mentions', [])
                content_data['mentions'] = [mention.get('screen_name', '') for mention in mentions]
                
                # Extract hashtags from entities
                hashtags = entities.get('hashtags', [])
                content_data['hashtags'] = [tag.get('text', '') for tag in hashtags]
                
                # Extract URLs from entities
                urls = entities.get('urls', [])
                content_data['urls'] = [url.get('expanded_url', url.get('url', '')) for url in urls]
                
                # Extract media from entities
                media = entities.get('media', [])
                media_urls = [item.get('media_url_https', item.get('media_url', '')) for item in media]
                content_data['media'].extend(media_urls)
            
            formatted_content = self.formatter.format_message_content(content_data, 'twitter')
            
            # Extract metadata
            metadata_data = {
                'posted_at': raw_message.get('created_at', ''),
                'url': f"https://twitter.com/{formatted_author['name']}/status/{tweet_id}",
                'reply_to': raw_message.get('in_reply_to_status_id_str', ''),
                'retweet_count': raw_message.get('retweet_count', 0),
                'like_count': raw_message.get('favorite_count', 0),
                'reply_count': raw_message.get('reply_count', 0),
                'quote_count': raw_message.get('quote_count', 0)
            }
            
            formatted_metadata = self.formatter.format_metadata(metadata_data, 'twitter')
            
            # Extract context (Twitter doesn't have channels/servers)
            context_data = {
                'channel': '',
                'server': '',
                'thread': raw_message.get('conversation_id', ''),
                'group': ''
            }
            
            formatted_context = self.formatter.format_context(context_data, 'twitter')
            
            return {
                'id': unified_id,
                'platform': 'twitter',
                'author': formatted_author,
                'content': formatted_content,
                'metadata': formatted_metadata,
                'context': formatted_context
            }
            
        except Exception as e:
            self.logger.error(f"Error converting Twitter message: {e}")
            return None
    
    def _convert_telegram_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Telegram message to unified format."""
        try:
            # Extract basic message info
            message_id = raw_message.get('id') or raw_message.get('message_id', '')
            if not message_id:
                self.logger.warning("Telegram message missing ID")
                return None
            
            # Generate unified ID
            unified_id = generate_message_id('telegram', str(message_id))
            
            # Extract author information
            author_data = raw_message.get('from_user', {}) or raw_message.get('sender', {})
            if not author_data:
                # Handle channel messages
                author_data = {
                    'name': raw_message.get('chat', {}).get('title', 'Channel'),
                    'id': raw_message.get('chat', {}).get('id', ''),
                    'username': raw_message.get('chat', {}).get('username', '')
                }
            
            formatted_author = self.formatter.format_author_info(author_data, 'telegram')
            
            # Extract content
            content_data = {
                'text': raw_message.get('text', '') or raw_message.get('message', ''),
                'html': raw_message.get('html', ''),
                'media': [],
                'attachments': []
            }
            
            # Handle Telegram media
            if raw_message.get('photo'):
                content_data['media'].append(raw_message['photo'])
            if raw_message.get('video'):
                content_data['media'].append(raw_message['video'])
            if raw_message.get('document'):
                content_data['attachments'].append(raw_message['document'])
            if raw_message.get('voice'):
                content_data['media'].append(raw_message['voice'])
            if raw_message.get('video_note'):
                content_data['media'].append(raw_message['video_note'])
            
            formatted_content = self.formatter.format_message_content(content_data, 'telegram')
            
            # Extract metadata
            metadata_data = {
                'posted_at': raw_message.get('date', ''),
                'url': '',  # Telegram doesn't have public URLs for all messages
                'reply_to': raw_message.get('reply_to_message_id', ''),
                'message_id': message_id,
                'chat_id': raw_message.get('chat', {}).get('id', ''),
                'forward_from': raw_message.get('forward_from', {}).get('id', '') if raw_message.get('forward_from') else ''
            }
            
            formatted_metadata = self.formatter.format_metadata(metadata_data, 'telegram')
            
            # Extract context
            chat_info = raw_message.get('chat', {})
            context_data = {
                'channel': chat_info.get('title', ''),
                'server': '',  # Telegram doesn't have servers
                'thread': raw_message.get('message_thread_id', ''),
                'group': chat_info.get('title', '') if chat_info.get('type') in ['group', 'supergroup'] else ''
            }
            
            formatted_context = self.formatter.format_context(context_data, 'telegram')
            
            return {
                'id': unified_id,
                'platform': 'telegram',
                'author': formatted_author,
                'content': formatted_content,
                'metadata': formatted_metadata,
                'context': formatted_context
            }
            
        except Exception as e:
            self.logger.error(f"Error converting Telegram message: {e}")
            return None
    
    def _convert_discord_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Discord message to unified format."""
        try:
            # Extract basic message info
            message_id = raw_message.get('id') or raw_message.get('message_id', '')
            if not message_id:
                self.logger.warning("Discord message missing ID")
                return None
            
            # Generate unified ID
            unified_id = generate_message_id('discord', str(message_id))
            
            # Extract author information
            author_data = raw_message.get('author', {}) or raw_message.get('user', {})
            if not author_data:
                self.logger.warning("Discord message missing author data")
                return None
            
            formatted_author = self.formatter.format_author_info(author_data, 'discord')
            
            # Extract content
            content_data = {
                'text': raw_message.get('content', ''),
                'html': raw_message.get('html', ''),
                'media': [],
                'attachments': raw_message.get('attachments', [])
            }
            
            # Handle Discord attachments
            attachments = raw_message.get('attachments', [])
            for attachment in attachments:
                if isinstance(attachment, dict):
                    url = attachment.get('url') or attachment.get('proxy_url')
                    if url:
                        content_data['media'].append(url)
            
            # Handle Discord embeds
            embeds = raw_message.get('embeds', [])
            for embed in embeds:
                if isinstance(embed, dict):
                    # Extract media from embeds
                    if embed.get('image', {}).get('url'):
                        content_data['media'].append(embed['image']['url'])
                    if embed.get('video', {}).get('url'):
                        content_data['media'].append(embed['video']['url'])
                    if embed.get('thumbnail', {}).get('url'):
                        content_data['media'].append(embed['thumbnail']['url'])
            
            formatted_content = self.formatter.format_message_content(content_data, 'discord')
            
            # Extract metadata
            metadata_data = {
                'posted_at': raw_message.get('timestamp', ''),
                'url': f"https://discord.com/channels/{raw_message.get('guild_id', '@me')}/{raw_message.get('channel_id', '')}/{message_id}",
                'reply_to': raw_message.get('referenced_message', {}).get('id', ''),
                'message_id': message_id,
                'channel_id': raw_message.get('channel_id', ''),
                'guild_id': raw_message.get('guild_id', ''),
                'thread_id': raw_message.get('thread_id', '')
            }
            
            # Handle Discord reactions
            reactions = {}
            discord_reactions = raw_message.get('reactions', [])
            for reaction in discord_reactions:
                if isinstance(reaction, dict):
                    emoji = reaction.get('emoji', {})
                    emoji_name = emoji.get('name', '') if emoji else ''
                    count = reaction.get('count', 0)
                    if emoji_name:
                        reactions[emoji_name] = count
            
            metadata_data['reactions'] = reactions
            formatted_metadata = self.formatter.format_metadata(metadata_data, 'discord')
            
            # Extract context
            context_data = {
                'channel': raw_message.get('channel_name', ''),
                'server': raw_message.get('guild_name', ''),
                'thread': raw_message.get('thread_name', ''),
                'group': ''  # Discord doesn't have groups in the same sense
            }
            
            formatted_context = self.formatter.format_context(context_data, 'discord')
            
            return {
                'id': unified_id,
                'platform': 'discord',
                'author': formatted_author,
                'content': formatted_content,
                'metadata': formatted_metadata,
                'context': formatted_context
            }
            
        except Exception as e:
            self.logger.error(f"Error converting Discord message: {e}")
            return None
