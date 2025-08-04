"""
Telegram scraper for TDXAgent using Telethon.

This module provides comprehensive Telegram data collection using the official
Telethon library, which is the most reliable and safe method for Telegram data access.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from pathlib import Path

from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from telethon.tl.types import (
    Channel, Chat, User, Message, MessageMediaPhoto, MessageMediaDocument,
    MessageMediaWebPage, PeerChannel, PeerChat, PeerUser
)

from scrapers.base_scraper import BaseScraper, ScrapingResult
from utils.logger import TDXLogger, log_async_function_call
from storage.optimized_message_converter import OptimizedMessageConverter


class TelegramScraper(BaseScraper):
    """
    Telegram data scraper using Telethon.
    
    Features:
    - Official Telegram API integration
    - Group and channel message collection
    - Media file handling
    - Group filtering and whitelisting
    - Rate limiting compliance
    - Session management
    """
    
    def __init__(self, config: Dict[str, Any], data_dir: Optional[str] = None):
        """
        Initialize Telegram scraper.
        
        Args:
            config: Telegram configuration dictionary
            data_dir: Data directory path (optional, for media storage)
        """
        super().__init__(config, "telegram")
        
        # Store data directory for media storage
        self.data_dir = data_dir or "TDXAgent_Data"
        
        # API credentials
        self.api_id = config.get('api_id')
        self.api_hash = config.get('api_hash')
        self.session_name = config.get('session_name', 'tdxagent_session')
        
        # Simple group filtering using direct group_whitelist
        self.group_whitelist = config.get('group_whitelist', [])
        self.case_sensitive = config.get('case_sensitive', False)
        self.partial_match = config.get('partial_match', True)
        
        # Convert to list if it's a set (for backward compatibility)
        if isinstance(self.group_whitelist, set):
            self.group_whitelist = list(self.group_whitelist)
        
        self.max_messages = config.get('max_messages', 1000)  # 向后兼容
        self.max_messages_per_group = config.get('max_messages_per_group', self.max_messages)
        self.max_total_messages = config.get('max_total_messages', 10000)
        self.enable_per_group_limit = config.get('enable_per_group_limit', True)
        
        self.logger.info(f"Telegram消息获取配置:")
        if self.enable_per_group_limit:
            self.logger.info(f"  单群组限制: {self.max_messages_per_group} 条消息")
            self.logger.info(f"  全局限制: {self.max_total_messages} 条消息")
        else:
            self.logger.info(f"  使用旧式全局限制: {self.max_messages} 条消息")
        
        # 用于跟踪总消息数
        self.total_messages_collected = 0
        
        self.logger.info(f"Telegram群组关键词过滤: {len(self.group_whitelist)} 个关键词")
        if self.group_whitelist:
            self.logger.info(f"关键词列表: {self.group_whitelist}")
            self.logger.info(f"区分大小写: {self.case_sensitive}, 部分匹配: {self.partial_match}")
        else:
            self.logger.info("无关键词配置，将获取所有群组")
        
        # Client instance
        self.client: Optional[TelegramClient] = None
        self.converter = OptimizedMessageConverter()
        
        # Validate configuration
        if not self.api_id or not self.api_hash:
            raise ValueError("Telegram API ID and hash are required")
        
        self.logger.info("Initialized Telegram scraper")
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Telegram using stored session or interactive login.
        
        Returns:
            True if authentication successful
        """
        try:
            # Get proxy configuration from config
            proxy = None
            
            # Access proxy config through the parent config manager
            # We need to get this from the TDXAgent instance
            # For now, we'll use a simple approach to get the proxy config
            from config.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            if config_manager.proxy.enabled:
                if config_manager.proxy.type == 'socks5':
                    proxy = {
                        'proxy_type': 'socks5',
                        'addr': config_manager.proxy.host,
                        'port': config_manager.proxy.port
                    }
                    if config_manager.proxy.username:
                        proxy['username'] = config_manager.proxy.username
                        proxy['password'] = config_manager.proxy.password
                    
                    self.logger.info(f"Using SOCKS5 proxy: {config_manager.proxy.host}:{config_manager.proxy.port}")
                elif config_manager.proxy.type in ['http', 'https']:
                    proxy = {
                        'proxy_type': 'http',
                        'addr': config_manager.proxy.host,
                        'port': config_manager.proxy.port
                    }
                    if config_manager.proxy.username:
                        proxy['username'] = config_manager.proxy.username
                        proxy['password'] = config_manager.proxy.password
                    
                    self.logger.info(f"Using HTTP proxy: {config_manager.proxy.host}:{config_manager.proxy.port}")
            
            # Create client with proxy support
            self.client = TelegramClient(
                self.session_name,
                self.api_id,
                self.api_hash,
                proxy=proxy
            )
            
            # Connect to Telegram
            await self.client.connect()
            
            # Check if already authorized
            if await self.client.is_user_authorized():
                self.logger.info("Already authenticated with existing session")
                self._is_authenticated = True
                return True
            
            # Interactive authentication
            self.logger.info("Starting interactive authentication...")
            
            # Clear the console and show instructions
            print("\n" + "="*60)
            print("🔑 Telegram 身份验证")
            print("="*60)
            
            # Request phone number
            print("请输入您的手机号码 (包含国家代码)")
            print("例如: +8613800138000")
            phone = input("手机号: ").strip()
            
            if not phone:
                raise ValueError("手机号不能为空")
            
            print(f"\n正在向 {phone} 发送验证码...")
            await self.client.send_code_request(phone)
            
            # Request verification code
            print("请检查短信并输入收到的验证码")
            code = input("验证码: ").strip()
            
            if not code:
                raise ValueError("验证码不能为空")
            
            try:
                await self.client.sign_in(phone, code)
            except SessionPasswordNeededError:
                # Two-factor authentication
                print("\n检测到两步验证，请输入您的密码")
                password = input("两步验证密码: ").strip()
                
                if not password:
                    raise ValueError("两步验证密码不能为空")
                    
                await self.client.sign_in(password=password)
            
            print("="*60)
            
            self.logger.info("Authentication successful")
            self._is_authenticated = True
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    async def scrape(self, hours_back: int = 12, **kwargs) -> ScrapingResult:
        """
        Scrape Telegram messages from groups and channels.
        
        Args:
            hours_back: Number of hours back to scrape
            **kwargs: Additional parameters
            
        Returns:
            ScrapingResult with collected messages
        """
        if not self.client or not self._is_authenticated:
            self.logger.error("Client not authenticated")
            return self.create_scraping_result([], ["Client not authenticated"])
        
        start_time, end_time = self.get_time_range(hours_back)
        
        # Convert UTC time to local time for display
        from datetime import timezone
        local_start = start_time.astimezone()  # Convert to local timezone
        local_end = end_time.astimezone()      # Convert to local timezone
        
        self.logger.info(f"Telegram时间范围: {local_start.strftime('%Y-%m-%d %H:%M:%S')} 到 {local_end.strftime('%Y-%m-%d %H:%M:%S')} (最近{hours_back}小时, 本地时间)")
        
        # 重置总消息计数
        self.total_messages_collected = 0
        
        messages = []
        errors = []
        
        try:
            # Get all dialogs (chats, groups, channels)
            dialogs = await self.client.get_dialogs()
            
            # Filter dialogs
            target_dialogs = self._filter_dialogs(dialogs)
            
            self.logger.info(f"Found {len(target_dialogs)} target dialogs to scrape")
            
            # Scrape each dialog
            for dialog in target_dialogs:
                try:
                    dialog_messages = await self._scrape_dialog(dialog, start_time, end_time)
                    messages.extend(dialog_messages)
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except FloodWaitError as e:
                    wait_time = e.seconds
                    self.logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                    
                    # Retry after waiting
                    try:
                        dialog_messages = await self._scrape_dialog(dialog, start_time, end_time)
                        messages.extend(dialog_messages)
                    except Exception as retry_error:
                        error_msg = f"Failed to scrape {dialog.name} after rate limit: {retry_error}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
                
                except Exception as e:
                    error_msg = f"Failed to scrape dialog {dialog.name}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Convert messages to unified format
            converted_messages = []
            for raw_message in messages:
                try:
                    converted = self.converter.convert_message(raw_message, 'telegram')
                    if converted:
                        converted_messages.append(converted)
                except Exception as e:
                    self.logger.warning(f"Failed to convert message: {e}")
            
            # 最终统计日志
            if self.enable_per_group_limit:
                self.logger.info(f"Telegram数据收集完成: 共获取 {len(converted_messages)} 条消息")
                self.logger.info(f"配置限制: 单群组最多 {self.max_messages_per_group} 条，全局最多 {self.max_total_messages} 条")
                if self.total_messages_collected >= self.max_total_messages:
                    self.logger.info(f"⚠️  已达到全局消息数量限制")
            else:
                self.logger.info(f"Scraped {len(converted_messages)} messages from Telegram (旧式全局限制: {self.max_messages})")
            
            return self.create_scraping_result(converted_messages, errors)
            
        except Exception as e:
            error_msg = f"Telegram scraping failed: {e}"
            self.logger.error(error_msg)
            return self.create_scraping_result([], [error_msg])
    
    def _filter_dialogs(self, dialogs) -> List[Any]:
        """
        Filter dialogs based on configured filtering mode.
        
        Args:
            dialogs: List of Telegram dialogs
            
        Returns:
            Filtered list of dialogs based on mode (exact, keywords, all)
        """
        target_dialogs = []
        filtered_count = 0
        
        for dialog in dialogs:
            # Only include groups and channels
            if isinstance(dialog.entity, (Channel, Chat)):
                group_name = dialog.name or ""
                should_include = False
                match_reason = ""
                
                # Simplified keyword-based filtering using group_whitelist
                if self.group_whitelist:
                    # Keywords matching using group_whitelist
                    self.logger.debug(f"🔍 检查群组 '{group_name}' 是否匹配关键词: {self.group_whitelist}")
                    for keyword in self.group_whitelist:
                        match_result = self._matches_keyword(group_name, keyword)
                        self.logger.debug(f"  - 关键词 '{keyword}': {'✅匹配' if match_result else '❌不匹配'}")
                        if match_result:
                            should_include = True
                            match_reason = f"关键词匹配: '{keyword}' in '{group_name}'"
                            break
                    if not should_include:
                        self.logger.debug(f"❌ 群组 '{group_name}' 无匹配关键词")
                else:
                    # No keywords configured, include all groups
                    should_include = True
                    match_reason = "无关键词配置，包含所有群组"
                
                if should_include:
                    target_dialogs.append(dialog)
                    self.logger.debug(f"✅ 包含群组: {group_name} ({match_reason})")
                else:
                    filtered_count += 1
                    self.logger.debug(f"❌ 过滤群组: {group_name} (不匹配过滤条件)")
        
        self.logger.info(f"群组过滤结果: 包含 {len(target_dialogs)} 个群组，过滤掉 {filtered_count} 个群组")
        return target_dialogs
    
    def _matches_keyword(self, group_name: str, keyword: str) -> bool:
        """
        Check if group name matches keyword based on configuration.
        
        Args:
            group_name: Name of the group
            keyword: Keyword to match
            
        Returns:
            True if matches
        """
        if not group_name or not keyword:
            return False
        
        # Prepare strings for comparison
        search_name = group_name if self.case_sensitive else group_name.lower()
        search_keyword = keyword if self.case_sensitive else keyword.lower()
        
        if self.partial_match:
            # Substring matching (关键字包含在群组名中)
            return search_keyword in search_name
        else:
            # Exact word matching
            import re
            # Use word boundaries to match whole words
            pattern = r'\b' + re.escape(search_keyword) + r'\b'
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return bool(re.search(pattern, group_name, flags))
    
    async def _scrape_dialog(self, dialog, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Scrape messages from a specific dialog.
        
        Args:
            dialog: Telegram dialog object
            start_time: Start time for message collection
            end_time: End time for message collection
            
        Returns:
            List of raw message dictionaries
        """
        messages = []
        messages_checked = 0
        messages_in_range = 0
        
        try:
            self.logger.debug(f"开始获取 {dialog.name} 的消息，时间范围: {start_time} - {end_time}")
            
            # 确定此群组的消息限制
            if self.enable_per_group_limit:
                group_limit = self.max_messages_per_group
                # 检查全局限制
                if self.total_messages_collected >= self.max_total_messages:
                    self.logger.info(f"已达到全局消息限制 {self.max_total_messages}，跳过群组 {dialog.name}")
                    return []
                # 调整群组限制以不超过全局剩余额度
                remaining_global = self.max_total_messages - self.total_messages_collected
                group_limit = min(group_limit, remaining_global)
            else:
                group_limit = self.max_messages
            
            self.logger.info(f"群组 {dialog.name}: 最多获取 {group_limit} 条消息")
            
            # Get messages from the dialog
            async for message in self.client.iter_messages(
                dialog.entity,
                offset_date=end_time,
                limit=group_limit
            ):
                messages_checked += 1
                
                # Check if message is within time range
                if message.date < start_time:
                    self.logger.debug(f"消息 {message.id} 时间 {message.date} 早于起始时间 {start_time}，停止获取")
                    break
                
                if message.date > end_time:
                    self.logger.debug(f"消息 {message.id} 时间 {message.date} 晚于结束时间 {end_time}，跳过")
                    continue
                
                messages_in_range += 1
                
                # Convert message to dictionary
                message_dict = await self._message_to_dict(message, dialog)
                if message_dict:
                    messages.append(message_dict)
                
                # 每100条消息记录一次进度
                if messages_checked % 100 == 0:
                    self.logger.debug(f"已检查 {messages_checked} 条消息，{messages_in_range} 条在时间范围内")
            
            collected_count = len(messages)
            self.logger.info(f"从 {dialog.name} 收集了 {collected_count} 条消息 (检查了{messages_checked}条，{messages_in_range}条在范围内)")
            
            # 更新总消息计数
            if self.enable_per_group_limit:
                self.total_messages_collected += collected_count
                remaining = self.max_total_messages - self.total_messages_collected
                self.logger.debug(f"全局消息统计: 已收集 {self.total_messages_collected}/{self.max_total_messages}，剩余 {remaining}")
            
        except Exception as e:
            self.logger.error(f"Failed to scrape dialog {dialog.name}: {e}")
            raise
        
        return messages
    
    async def _message_to_dict(self, message: Message, dialog) -> Optional[Dict[str, Any]]:
        """
        Convert Telegram message to dictionary format.
        
        Args:
            message: Telegram message object
            dialog: Dialog containing the message
            
        Returns:
            Message dictionary or None if conversion fails
        """
        try:
            # Basic message info
            message_dict = {
                'id': message.id,
                'message_id': message.id,
                'text': message.text or '',
                'date': message.date.isoformat(),
                'timestamp': message.date.timestamp(),
            }
            
            # Author information
            if message.sender:
                if isinstance(message.sender, User):
                    message_dict['from_user'] = {
                        'id': message.sender.id,
                        'name': message.sender.username or f"user_{message.sender.id}",
                        'first_name': message.sender.first_name or '',
                        'last_name': message.sender.last_name or '',
                        'username': message.sender.username or ''
                    }
                elif isinstance(message.sender, Channel):
                    message_dict['from_user'] = {
                        'id': message.sender.id,
                        'name': message.sender.title or f"channel_{message.sender.id}",
                        'username': message.sender.username or ''
                    }
            
            # Chat information
            if isinstance(dialog.entity, Channel):
                message_dict['chat'] = {
                    'id': dialog.entity.id,
                    'title': dialog.entity.title,
                    'username': dialog.entity.username or '',
                    'type': 'channel' if dialog.entity.broadcast else 'supergroup'
                }
            elif isinstance(dialog.entity, Chat):
                message_dict['chat'] = {
                    'id': dialog.entity.id,
                    'title': dialog.entity.title,
                    'type': 'group'
                }
            
            # Reply information
            if message.reply_to:
                message_dict['reply_to_message_id'] = message.reply_to.reply_to_msg_id
            
            # Forward information
            if message.forward:
                forward_info = {}
                if message.forward.from_id:
                    if isinstance(message.forward.from_id, PeerUser):
                        forward_info['from_user_id'] = message.forward.from_id.user_id
                    elif isinstance(message.forward.from_id, PeerChannel):
                        forward_info['from_channel_id'] = message.forward.from_id.channel_id
                
                if message.forward.date:
                    forward_info['date'] = message.forward.date.isoformat()
                
                message_dict['forward_from'] = forward_info
            
            # Media information
            media_urls = []
            if message.media:
                if isinstance(message.media, MessageMediaPhoto):
                    # Photo media
                    photo_path = await self._download_media(message, 'photo')
                    if photo_path:
                        media_urls.append(photo_path)
                
                elif isinstance(message.media, MessageMediaDocument):
                    # Document/video/audio media
                    doc_path = await self._download_media(message, 'document')
                    if doc_path:
                        media_urls.append(doc_path)
                
                elif isinstance(message.media, MessageMediaWebPage):
                    # Web page preview
                    if message.media.webpage.url:
                        media_urls.append(message.media.webpage.url)
            
            if media_urls:
                message_dict['media_urls'] = media_urls
            
            return message_dict
            
        except Exception as e:
            self.logger.warning(f"Failed to convert message {message.id}: {e}")
            return None
    
    async def _download_media(self, message: Message, media_type: str) -> Optional[str]:
        """
        Download media file from message.
        
        Args:
            message: Telegram message with media
            media_type: Type of media ('photo', 'document')
            
        Returns:
            Local file path or None if download fails
        """
        try:
            # Create media directory using configured data directory
            media_dir = Path(self.data_dir) / "data" / "telegram" / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            
            # Download media
            file_path = await self.client.download_media(
                message,
                file=media_dir
            )
            
            if file_path:
                return str(file_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to download {media_type} from message {message.id}: {e}")
        
        return None
    
    async def cleanup(self) -> None:
        """Clean up Telegram client connection."""
        if self.client:
            try:
                await self.client.disconnect()
                self.logger.info("Telegram client disconnected")
            except Exception as e:
                self.logger.warning(f"Error disconnecting Telegram client: {e}")
    
    async def get_group_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available groups and channels with filtering status.
        
        Returns:
            List of group/channel information with match status
        """
        if not self.client or not self._is_authenticated:
            return []
        
        groups = []
        
        try:
            dialogs = await self.client.get_dialogs()
            
            for dialog in dialogs:
                if isinstance(dialog.entity, (Channel, Chat)):
                    group_name = dialog.name or ""
                    
                    # Determine if group would be included based on current filter settings
                    would_include = False
                    match_info = ""
                    
                    # Simplified keyword-based filtering using group_whitelist
                    if self.group_whitelist:
                        # Check if group name matches any keyword in whitelist
                        for keyword in self.group_whitelist:
                            if self._matches_keyword(group_name, keyword):
                                would_include = True
                                match_info = f"关键词匹配: '{keyword}'"
                                break
                        if not would_include:
                            match_info = "无关键词匹配"
                    else:
                        would_include = True
                        match_info = "无关键词配置，包含所有群组"
                    
                    group_info = {
                        'id': dialog.entity.id,
                        'name': group_name,
                        'type': 'channel' if isinstance(dialog.entity, Channel) and dialog.entity.broadcast else 'group',
                        'username': getattr(dialog.entity, 'username', '') or '',
                        'member_count': getattr(dialog.entity, 'participants_count', 0),
                        'would_include': would_include,
                        'match_info': match_info,
                        # Legacy field for backward compatibility
                        'in_whitelist': would_include
                    }
                    groups.append(group_info)
            
        except Exception as e:
            self.logger.error(f"Failed to get group list: {e}")
        
        return groups
    
    async def add_to_whitelist(self, group_name: str) -> bool:
        """
        Add a group to the whitelist.
        
        Args:
            group_name: Name of the group to add to whitelist
            
        Returns:
            True if added successfully
        """
        if group_name not in self.group_whitelist:
            self.group_whitelist.append(group_name)
            self.logger.info(f"Added '{group_name}' to whitelist")
            return True
        else:
            self.logger.info(f"'{group_name}' already in whitelist")
            return True
    
    async def remove_from_whitelist(self, group_name: str) -> bool:
        """
        Remove a group from the whitelist.
        
        Args:
            group_name: Name of the group to remove from whitelist
            
        Returns:
            True if removed successfully
        """
        if group_name in self.group_whitelist:
            self.group_whitelist.remove(group_name)
            self.logger.info(f"Removed '{group_name}' from whitelist")
            return True
        else:
            self.logger.warning(f"'{group_name}' not found in whitelist")
            return False
