"""
JSONL storage system for TDXAgent.

This module provides efficient storage and retrieval of message data using
JSON Lines format, which is ideal for streaming data and large datasets.
"""

import asyncio
import json
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from datetime import datetime, date, timezone
import logging
from contextlib import asynccontextmanager
import fcntl
import os

from utils.logger import TDXLogger
from utils.helpers import ensure_directory_async, format_timestamp, sanitize_filename
from utils.validators import validate_message


class JSONLStorage:
    """
    JSONL-based storage system for TDXAgent messages.
    
    Features:
    - Efficient streaming read/write operations
    - Date-based file organization
    - Data integrity validation
    - Concurrent access handling
    - Automatic backup and recovery
    """
    
    def __init__(self, base_directory: Union[str, Path] = "TDXAgent_Data"):
        """
        Initialize JSONL storage.
        
        Args:
            base_directory: Base directory for data storage
        """
        self.base_directory = Path(base_directory)
        self.logger = TDXLogger.get_logger("tdxagent.storage")
        
        # Storage structure: base_directory/platform/YYYY-MM-DD.jsonl
        self.data_directory = self.base_directory / "data"
        
        # File locks for concurrent access
        self._file_locks: Dict[str, asyncio.Lock] = {}
    
    def _extract_date_from_posted_at(self, posted_at: str) -> Optional[date]:
        """
        Extract date from posted_at timestamp.
        
        Args:
            posted_at: ISO timestamp string
            
        Returns:
            Date object or None if parsing fails
        """
        try:
            # Parse ISO format timestamp and extract date
            dt = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
            return dt.date()
        except (ValueError, AttributeError) as e:
            self.logger.warning(f"Failed to parse posted_at timestamp '{posted_at}': {e}")
            return None
    
    async def get_messages_by_time_range(self, platform: str,
                                       start_time: datetime,
                                       end_time: datetime,
                                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages by exact time range (more precise than date-based filtering).
        
        Args:
            platform: Platform name
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            limit: Maximum number of messages
            
        Returns:
            List of messages within the time range
        """
        # 🎯 改进：由于新数据按抓取日期存储，只需要扫描精确的日期范围
        # 不再需要扩展日期范围，避免扫描不必要的文件
        start_date = start_time.date()
        end_date = end_time.date()
        
        # 对于现有旧数据的向后兼容：仍然扩展范围以确保不遗漏
        # 但新数据将主要集中在精确的日期范围内
        extended_start = date.fromordinal(max(1, start_date.toordinal() - 1))
        extended_end = date.fromordinal(end_date.toordinal() + 1)
        
        # 打印数据路径信息用于调试
        platform_dir = self.data_directory / platform
        self.logger.info(f"读取 {platform} 数据路径: {platform_dir}")
        self.logger.info(f"时间范围: {start_time} 到 {end_time}")
        self.logger.info(f"优先扫描精确日期: {start_date} 到 {end_date}")
        self.logger.info(f"向后兼容扫描范围: {extended_start} 到 {extended_end}")
        
        # 列出实际存在的文件
        if platform_dir.exists():
            existing_files = list(platform_dir.glob("*.jsonl"))
            self.logger.info(f"找到 {len(existing_files)} 个数据文件: {[f.name for f in existing_files]}")
        else:
            self.logger.warning(f"平台目录不存在: {platform_dir}")
        
        messages = []
        count = 0
        
        # Read all messages in the extended date range
        async for message in self.read_messages(platform, extended_start, extended_end):
            if limit and count >= limit:
                break
                
            # Parse posted_at timestamp for precise filtering
            posted_at_str = message.get('metadata', {}).get('posted_at')
            if posted_at_str:
                try:
                    posted_at = datetime.fromisoformat(posted_at_str.replace('Z', '+00:00'))
                    
                    # 确保时区一致性进行比较
                    # 如果 start_time 和 end_time 有时区信息，确保 posted_at 也有
                    if start_time.tzinfo is not None and posted_at.tzinfo is None:
                        # start_time 有时区但 posted_at 没有，假设 posted_at 是 UTC
                        posted_at = posted_at.replace(tzinfo=timezone.utc)
                    elif start_time.tzinfo is None and posted_at.tzinfo is not None:
                        # start_time 没有时区但 posted_at 有，转换 posted_at 为无时区
                        posted_at = posted_at.replace(tzinfo=None)
                    
                    # Check if message is within the exact time range
                    if start_time <= posted_at <= end_time:
                        messages.append(message)
                        count += 1
                except (ValueError, AttributeError) as e:
                    # If timestamp parsing fails, skip this message
                    self.logger.debug(f"Failed to parse posted_at for message {message.get('id', 'unknown')}: {e}")
                    continue
        
        # Sort by posted_at timestamp (newest first)
        messages.sort(key=lambda m: m.get('metadata', {}).get('posted_at', ''), reverse=True)
        
        self.logger.info(f"时间范围筛选完成: 共收集到 {len(messages)} 条 {platform} 消息")
        
        return messages
    
    async def initialize(self) -> None:
        """Initialize storage directories."""
        await ensure_directory_async(self.data_directory)
        
        # Create platform directories
        platforms = ['twitter', 'telegram', 'discord']
        for platform in platforms:
            await ensure_directory_async(self.data_directory / platform)
            await ensure_directory_async(self.data_directory / platform / "media")
        
        self.logger.info("Storage directories initialized")
    
    def _get_file_path(self, platform: str, date_obj: Optional[date] = None) -> Path:
        """
        Get the file path for a platform and date.
        
        Args:
            platform: Platform name
            date_obj: Date object (defaults to today)
            
        Returns:
            Path to the JSONL file
        """
        if date_obj is None:
            date_obj = date.today()
        
        filename = f"{date_obj.isoformat()}.jsonl"
        return self.data_directory / platform / filename
    
    def _get_file_lock(self, file_path: Path) -> asyncio.Lock:
        """Get or create a lock for a specific file."""
        file_key = str(file_path)
        if file_key not in self._file_locks:
            self._file_locks[file_key] = asyncio.Lock()
        return self._file_locks[file_key]
    
    @asynccontextmanager
    async def _file_lock_context(self, file_path: Path):
        """Context manager for file locking."""
        lock = self._get_file_lock(file_path)
        async with lock:
            yield
    
    async def store_message(self, platform: str, message: Dict[str, Any], 
                           validate: bool = True) -> bool:
        """
        Store a single message.
        
        Args:
            platform: Platform name
            message: Message data
            validate: Whether to validate the message format
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Validate message if requested
            if validate and not validate_message(message):
                self.logger.warning("Message validation failed, skipping storage")
                return False
            
            # Add storage metadata with version upgrade
            message['_storage'] = {
                'stored_at': datetime.now().isoformat(),
                'platform': platform,
                'version': '2.0',  # 🎯 版本升级：v2.0使用抓取时间存储
                'storage_strategy': 'scraping_time'  # 标识新的存储策略
            }
            
            # 🎯 改进：按抓取时间组织文件，而不是发布时间
            # 这样analyze时只需要扫描精确的时间范围，不会遗漏For You页面的跨时间推文
            date_obj = date.today()  # 使用抓取日期
            
            # 保留原始逻辑作为向后兼容fallback（用于现有数据读取）
            posted_at = message.get('metadata', {}).get('posted_at')
            if posted_at:
                # 保存原始发布时间到metadata中，用于时间过滤
                pass  # posted_at already in metadata
            
            file_path = self._get_file_path(platform, date_obj)
            
            # Ensure directory exists
            await ensure_directory_async(file_path.parent)
            
            # Write message with file locking
            async with self._file_lock_context(file_path):
                async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                    line = json.dumps(message, ensure_ascii=False) + '\n'
                    await f.write(line)
            
            self.logger.debug(f"Stored message {message.get('id', 'unknown')} to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store message: {e}")
            return False
    
    async def store_messages(self, platform: str, messages: List[Dict[str, Any]], 
                           validate: bool = True) -> tuple[int, int]:
        """
        Store multiple messages efficiently.
        
        Args:
            platform: Platform name
            messages: List of message data
            validate: Whether to validate message formats
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not messages:
            return 0, 0
        
        successful = 0
        failed = 0
        
        # 🎯 改进：所有消息按抓取日期组织，不再按发布时间分散存储
        # Group messages by scraping date for efficient writing
        scraping_date = date.today()  # 使用抓取日期
        messages_by_date: Dict[date, List[Dict[str, Any]]] = {scraping_date: []}
        
        for message in messages:
            # Validate if requested
            if validate and not validate_message(message):
                failed += 1
                continue
            
            # Add storage metadata with version upgrade
            message['_storage'] = {
                'stored_at': datetime.now().isoformat(),
                'platform': platform,
                'version': '2.0',  # 🎯 版本升级：v2.0使用抓取时间存储
                'storage_strategy': 'scraping_time'  # 标识新的存储策略
            }
            
            # 所有消息都存储到同一个抓取日期文件中
            # posted_at时间仍然保留在metadata中用于后续的时间过滤
            messages_by_date[scraping_date].append(message)
        
        # Write messages grouped by date
        for date_obj, date_messages in messages_by_date.items():
            try:
                file_path = self._get_file_path(platform, date_obj)
                await ensure_directory_async(file_path.parent)
                
                async with self._file_lock_context(file_path):
                    async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                        for message in date_messages:
                            line = json.dumps(message, ensure_ascii=False) + '\n'
                            await f.write(line)
                
                successful += len(date_messages)
                self.logger.info(f"Stored {len(date_messages)} messages to {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to store messages for {date_obj}: {e}")
                failed += len(date_messages)
        
        self.logger.info(f"Batch storage complete: {successful} successful, {failed} failed")
        return successful, failed
    
    async def read_messages(self, platform: str, 
                           start_date: Optional[date] = None,
                           end_date: Optional[date] = None,
                           limit: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Read messages from storage.
        
        Args:
            platform: Platform name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Maximum number of messages to read
            
        Yields:
            Message dictionaries
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date
        
        count = 0
        current_date = start_date
        
        while current_date <= end_date:
            if limit and count >= limit:
                break
            
            file_path = self._get_file_path(platform, current_date)
            
            if file_path.exists():
                try:
                    async with self._file_lock_context(file_path):
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            async for line in f:
                                if limit and count >= limit:
                                    break
                                
                                line = line.strip()
                                if line:
                                    try:
                                        message = json.loads(line)
                                        yield message
                                        count += 1
                                    except json.JSONDecodeError as e:
                                        self.logger.warning(f"Failed to parse line in {file_path}: {e}")
                
                except Exception as e:
                    self.logger.error(f"Failed to read from {file_path}: {e}")
            
            # Move to next date
            current_date = date.fromordinal(current_date.toordinal() + 1)
    
    async def get_messages_list(self, platform: str,
                               start_date: Optional[date] = None,
                               end_date: Optional[date] = None,
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages as a list (convenience method).
        
        Args:
            platform: Platform name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Maximum number of messages
            
        Returns:
            List of messages
        """
        messages = []
        async for message in self.read_messages(platform, start_date, end_date, limit):
            messages.append(message)
        return messages
    
    async def count_messages(self, platform: str,
                            start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> int:
        """
        Count messages in storage.
        
        Args:
            platform: Platform name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Number of messages
        """
        count = 0
        async for _ in self.read_messages(platform, start_date, end_date):
            count += 1
        return count
    
    async def get_available_dates(self, platform: str) -> List[date]:
        """
        Get list of dates with available data for a platform.
        
        Args:
            platform: Platform name
            
        Returns:
            List of dates with data
        """
        platform_dir = self.data_directory / platform
        dates = []
        
        if platform_dir.exists():
            for file_path in platform_dir.glob("*.jsonl"):
                try:
                    date_str = file_path.stem
                    date_obj = date.fromisoformat(date_str)
                    dates.append(date_obj)
                except ValueError:
                    self.logger.warning(f"Invalid date format in filename: {file_path}")
        
        return sorted(dates)
    
    async def delete_messages(self, platform: str, 
                             start_date: Optional[date] = None,
                             end_date: Optional[date] = None) -> int:
        """
        Delete messages from storage.
        
        Args:
            platform: Platform name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Number of files deleted
        """
        if start_date is None and end_date is None:
            self.logger.warning("Refusing to delete all data without date range")
            return 0
        
        if start_date is None:
            start_date = end_date
        if end_date is None:
            end_date = start_date
        
        deleted_count = 0
        current_date = start_date
        
        while current_date <= end_date:
            file_path = self._get_file_path(platform, current_date)
            
            if file_path.exists():
                try:
                    async with self._file_lock_context(file_path):
                        file_path.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {file_path}: {e}")
            
            current_date = date.fromordinal(current_date.toordinal() + 1)
        
        return deleted_count
    
    async def backup_data(self, platform: str, backup_directory: Union[str, Path]) -> bool:
        """
        Create a backup of platform data.
        
        Args:
            platform: Platform name
            backup_directory: Directory to store backup
            
        Returns:
            True if backup successful
        """
        try:
            backup_dir = Path(backup_directory)
            await ensure_directory_async(backup_dir)
            
            platform_dir = self.data_directory / platform
            if not platform_dir.exists():
                self.logger.warning(f"No data found for platform {platform}")
                return False
            
            # Create timestamped backup directory
            timestamp = format_timestamp(format_type="filename")
            backup_platform_dir = backup_dir / f"{platform}_{timestamp}"
            await ensure_directory_async(backup_platform_dir)
            
            # Copy all JSONL files
            copied_files = 0
            for file_path in platform_dir.glob("*.jsonl"):
                backup_file = backup_platform_dir / file_path.name
                
                async with aiofiles.open(file_path, 'rb') as src:
                    async with aiofiles.open(backup_file, 'wb') as dst:
                        content = await src.read()
                        await dst.write(content)
                
                copied_files += 1
            
            self.logger.info(f"Backup completed: {copied_files} files copied to {backup_platform_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'platforms': {},
            'total_files': 0,
            'total_size_bytes': 0
        }
        
        for platform_dir in self.data_directory.iterdir():
            if platform_dir.is_dir() and platform_dir.name != 'media':
                platform_name = platform_dir.name
                platform_stats = {
                    'files': 0,
                    'size_bytes': 0,
                    'date_range': None,
                    'message_count': 0
                }
                
                dates = []
                for file_path in platform_dir.glob("*.jsonl"):
                    if file_path.is_file():
                        platform_stats['files'] += 1
                        platform_stats['size_bytes'] += file_path.stat().st_size
                        
                        # Extract date from filename
                        try:
                            date_obj = date.fromisoformat(file_path.stem)
                            dates.append(date_obj)
                        except ValueError:
                            pass
                
                if dates:
                    platform_stats['date_range'] = {
                        'start': min(dates).isoformat(),
                        'end': max(dates).isoformat()
                    }
                
                # Count messages (this could be expensive for large datasets)
                platform_stats['message_count'] = await self.count_messages(platform_name)
                
                stats['platforms'][platform_name] = platform_stats
                stats['total_files'] += platform_stats['files']
                stats['total_size_bytes'] += platform_stats['size_bytes']
        
        return stats
