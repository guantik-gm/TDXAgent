"""
Discord scraper for TDXAgent with safe and experimental modes.

This module provides Discord data collection with two modes:
1. Safe mode: Uses official Discord data exports
2. Experimental mode: Direct API access (high risk of account ban)
"""

import json
import zipfile
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from scrapers.base_scraper import BaseScraper, ScrapingResult
from utils.logger import TDXLogger, log_async_function_call
from storage.message_converter import MessageConverter
from utils.helpers import ensure_directory


class DiscordSafeProcessor:
    """
    Safe Discord data processor for official exports.
    
    This processor handles Discord's official data export files,
    which is the safest way to access Discord data without risking account bans.
    """
    
    def __init__(self, export_directory: str = "discord_exports"):
        """
        Initialize Discord safe processor.
        
        Args:
            export_directory: Directory containing Discord export files
        """
        self.export_directory = Path(export_directory)
        self.logger = TDXLogger.get_logger("tdxagent.scrapers.discord.safe")
        self.converter = MessageConverter()
    
    async def process_export(self, export_file: str, hours_back: int = 12) -> List[Dict[str, Any]]:
        """
        Process Discord data export file.
        
        Args:
            export_file: Path to Discord export ZIP file
            hours_back: Number of hours back to include messages
            
        Returns:
            List of processed messages
        """
        export_path = Path(export_file)
        
        if not export_path.exists():
            raise FileNotFoundError(f"Export file not found: {export_path}")
        
        messages = []
        start_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            with zipfile.ZipFile(export_path, 'r') as zip_file:
                # Extract to temporary directory
                temp_dir = self.export_directory / "temp_extract"
                await ensure_directory(temp_dir)
                
                zip_file.extractall(temp_dir)
                
                # Process messages
                messages_dir = temp_dir / "messages"
                if messages_dir.exists():
                    messages = await self._process_messages_directory(messages_dir, start_time)
                
                # Clean up temporary files
                await self._cleanup_temp_directory(temp_dir)
        
        except Exception as e:
            self.logger.error(f"Failed to process Discord export: {e}")
            raise
        
        return messages
    
    async def _process_messages_directory(self, messages_dir: Path, start_time: datetime) -> List[Dict[str, Any]]:
        """Process messages from extracted directory."""
        messages = []
        
        # Discord exports organize messages by channel
        for channel_dir in messages_dir.iterdir():
            if channel_dir.is_dir():
                channel_messages = await self._process_channel_directory(channel_dir, start_time)
                messages.extend(channel_messages)
        
        return messages
    
    async def _process_channel_directory(self, channel_dir: Path, start_time: datetime) -> List[Dict[str, Any]]:
        """Process messages from a specific channel directory."""
        messages = []
        
        # Look for messages.csv or messages.json files
        messages_csv = channel_dir / "messages.csv"
        messages_json = channel_dir / "messages.json"
        
        if messages_csv.exists():
            messages = await self._process_csv_messages(messages_csv, start_time, channel_dir.name)
        elif messages_json.exists():
            messages = await self._process_json_messages(messages_json, start_time, channel_dir.name)
        
        return messages
    
    async def _process_csv_messages(self, csv_file: Path, start_time: datetime, channel_name: str) -> List[Dict[str, Any]]:
        """Process messages from CSV file."""
        messages = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse timestamp
                        timestamp_str = row.get('Timestamp', '')
                        if timestamp_str:
                            message_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if message_time < start_time:
                                continue
                        
                        # Create message dictionary
                        message_dict = {
                            'id': row.get('ID', ''),
                            'content': row.get('Contents', ''),
                            'timestamp': timestamp_str,
                            'author': {
                                'id': row.get('Author ID', ''),
                                'name': row.get('Author', ''),
                            },
                            'channel_name': channel_name,
                            'attachments': self._parse_attachments(row.get('Attachments', '')),
                            'reactions': self._parse_reactions(row.get('Reactions', ''))
                        }
                        
                        messages.append(message_dict)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process CSV row: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {csv_file}: {e}")
        
        return messages
    
    async def _process_json_messages(self, json_file: Path, start_time: datetime, channel_name: str) -> List[Dict[str, Any]]:
        """Process messages from JSON file."""
        messages = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Discord JSON exports have different structures
            if isinstance(data, list):
                message_list = data
            elif isinstance(data, dict) and 'messages' in data:
                message_list = data['messages']
            else:
                self.logger.warning(f"Unknown JSON structure in {json_file}")
                return messages
            
            for message_data in message_list:
                try:
                    # Parse timestamp
                    timestamp_str = message_data.get('timestamp', '')
                    if timestamp_str:
                        message_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if message_time < start_time:
                            continue
                    
                    # Create message dictionary
                    message_dict = {
                        'id': message_data.get('id', ''),
                        'content': message_data.get('content', ''),
                        'timestamp': timestamp_str,
                        'author': message_data.get('author', {}),
                        'channel_name': channel_name,
                        'attachments': message_data.get('attachments', []),
                        'embeds': message_data.get('embeds', []),
                        'reactions': message_data.get('reactions', [])
                    }
                    
                    messages.append(message_dict)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process JSON message: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {json_file}: {e}")
        
        return messages
    
    def _parse_attachments(self, attachments_str: str) -> List[str]:
        """Parse attachments from CSV string."""
        if not attachments_str:
            return []
        
        # Simple parsing - attachments are usually URLs separated by commas
        return [url.strip() for url in attachments_str.split(',') if url.strip()]
    
    def _parse_reactions(self, reactions_str: str) -> Dict[str, int]:
        """Parse reactions from CSV string."""
        reactions = {}
        
        if not reactions_str:
            return reactions
        
        # Simple parsing - format might be "👍:5,❤️:3"
        try:
            for reaction in reactions_str.split(','):
                if ':' in reaction:
                    emoji, count = reaction.strip().split(':', 1)
                    reactions[emoji] = int(count)
        except Exception as e:
            self.logger.warning(f"Failed to parse reactions: {e}")
        
        return reactions
    
    async def _cleanup_temp_directory(self, temp_dir: Path) -> None:
        """Clean up temporary extraction directory."""
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp directory: {e}")
    
    def get_export_guide(self) -> str:
        """Get instructions for Discord data export."""
        return """
Discord 数据导出步骤：

1. 打开 Discord 应用或网页版
2. 点击用户设置（齿轮图标）
3. 在左侧菜单中选择"隐私与安全"
4. 向下滚动找到"请求我的所有数据"
5. 点击"请求数据"按钮
6. 确认请求（可能需要验证邮箱）
7. 等待 Discord 处理请求（通常需要 1-30 天）
8. 收到邮件通知后，下载数据包
9. 将下载的 ZIP 文件放入 discord_exports 目录
10. 在 TDXAgent 中使用安全模式处理数据

注意：
- 数据导出是完全合法和安全的
- 不会违反 Discord 服务条款
- 不会导致账户封禁
- 数据包含您的所有消息历史
"""


class DiscordScraper(BaseScraper):
    """
    Discord scraper with safe and experimental modes.
    
    Safe mode: Uses official Discord data exports (recommended)
    Experimental mode: Direct API access (high risk)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Discord scraper.
        
        Args:
            config: Discord configuration dictionary
        """
        super().__init__(config, "discord")
        
        self.mode = config.get('mode', 'safe')
        self.export_path = config.get('export_path', 'discord_exports')
        
        # Initialize processors
        self.safe_processor = DiscordSafeProcessor(self.export_path)
        
        # Experimental mode configuration
        self.experimental_config = config.get('experimental', {})
        
        self.converter = MessageConverter()
        
        self.logger.info(f"Initialized Discord scraper in {self.mode} mode")
    
    async def authenticate(self) -> bool:
        """
        Authenticate Discord scraper.
        
        For safe mode, no authentication needed.
        For experimental mode, validate token.
        """
        if self.mode == 'safe':
            # Safe mode doesn't require authentication
            self._is_authenticated = True
            return True
        
        elif self.mode == 'experimental':
            # Show warning for experimental mode
            if not self._show_experimental_warning():
                return False
            
            # Validate token
            token = self.experimental_config.get('token', '')
            if not token:
                self.logger.error("Discord token required for experimental mode")
                return False
            
            # TODO: Implement experimental mode authentication
            self.logger.warning("Experimental mode not yet implemented")
            return False
        
        return False
    
    def _show_experimental_warning(self) -> bool:
        """Show warning for experimental mode and get user confirmation."""
        warning = """
⚠️  严重警告 ⚠️

Discord 实验模式存在以下风险：
1. 违反 Discord 服务条款
2. 可能导致账户永久封禁
3. Discord 检测算法日益严格
4. 可能面临法律后果

强烈建议使用安全模式（官方数据导出）。

如果您完全理解风险并愿意承担后果，请输入 'I_UNDERSTAND_THE_RISKS'：
"""
        
        try:
            user_input = input(warning).strip()
            return user_input == 'I_UNDERSTAND_THE_RISKS'
        except (EOFError, KeyboardInterrupt):
            return False
    
    async def scrape(self, hours_back: int = 12, **kwargs) -> ScrapingResult:
        """
        Scrape Discord data based on configured mode.
        
        Args:
            hours_back: Number of hours back to scrape
            **kwargs: Additional parameters
            
        Returns:
            ScrapingResult with collected messages
        """
        if self.mode == 'safe':
            return await self._scrape_safe_mode(hours_back, **kwargs)
        elif self.mode == 'experimental':
            return await self._scrape_experimental_mode(hours_back, **kwargs)
        else:
            error_msg = f"Unknown Discord mode: {self.mode}"
            self.logger.error(error_msg)
            return self.create_scraping_result([], [error_msg])
    
    async def _scrape_safe_mode(self, hours_back: int, **kwargs) -> ScrapingResult:
        """Scrape using safe mode (official exports)."""
        try:
            # Look for export files
            export_files = list(Path(self.export_path).glob("*.zip"))
            
            if not export_files:
                error_msg = f"No Discord export files found in {self.export_path}"
                self.logger.error(error_msg)
                
                # Provide guidance
                guide = self.safe_processor.get_export_guide()
                self.logger.info(guide)
                
                return self.create_scraping_result([], [error_msg])
            
            # Process the most recent export file
            latest_export = max(export_files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"Processing Discord export: {latest_export}")
            
            # Process messages
            raw_messages = await self.safe_processor.process_export(str(latest_export), hours_back)
            
            # Convert to unified format
            converted_messages = []
            for raw_message in raw_messages:
                try:
                    converted = self.converter.convert_message(raw_message, 'discord')
                    if converted:
                        converted_messages.append(converted)
                except Exception as e:
                    self.logger.warning(f"Failed to convert Discord message: {e}")
            
            self.logger.info(f"Processed {len(converted_messages)} Discord messages")
            return self.create_scraping_result(converted_messages, [])
            
        except Exception as e:
            error_msg = f"Discord safe mode scraping failed: {e}"
            self.logger.error(error_msg)
            return self.create_scraping_result([], [error_msg])
    
    async def _scrape_experimental_mode(self, hours_back: int, **kwargs) -> ScrapingResult:
        """Scrape using experimental mode (direct API)."""
        # TODO: Implement experimental mode
        error_msg = "Discord experimental mode not yet implemented"
        self.logger.error(error_msg)
        return self.create_scraping_result([], [error_msg])
    
    async def cleanup(self) -> None:
        """Clean up Discord scraper resources."""
        # No cleanup needed for safe mode
        if self.mode == 'experimental':
            # TODO: Cleanup experimental mode resources
            pass
        
        self.logger.info("Discord scraper cleanup completed")
    
    def get_export_instructions(self) -> str:
        """Get Discord data export instructions."""
        return self.safe_processor.get_export_guide()
