"""
Base scraper class for TDXAgent platform data collection.

This module provides the abstract base class that all platform-specific
scrapers inherit from, ensuring consistent interface and common functionality.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path

from utils.logger import TDXLogger, log_async_function_call
from utils.validators import validate_message


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    platform: str
    messages: List[Dict[str, Any]]
    total_count: int
    success_count: int
    error_count: int
    start_time: datetime
    end_time: datetime
    errors: List[str]
    
    @property
    def duration(self) -> float:
        """Get the duration of the scraping operation in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Get the success rate as a percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100


class BaseScraper(ABC):
    """
    Abstract base class for all platform scrapers.
    
    This class provides common functionality and enforces a consistent
    interface for all platform-specific scrapers.
    
    Features:
    - Standardized message format
    - Error handling and retry logic
    - Rate limiting
    - Progress tracking
    - Data validation
    """
    
    def __init__(self, config: Dict[str, Any], platform_name: str):
        """
        Initialize the base scraper.
        
        Args:
            config: Platform-specific configuration
            platform_name: Name of the platform (e.g., 'twitter', 'telegram')
        """
        self.config = config
        self.platform_name = platform_name
        self.logger = TDXLogger.get_logger(f"tdxagent.scrapers.{platform_name}")
        
        # Scraping state
        self._is_authenticated = False
        self._last_request_time = 0.0
        self._request_count = 0
        self._session_start_time: Optional[datetime] = None
        
        # Rate limiting
        self.min_delay = config.get('min_delay', 1.0)
        self.max_requests_per_minute = config.get('max_requests_per_minute', 60)
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5.0)
        
        self.logger.info(f"Initialized {platform_name} scraper")
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the platform.
        
        Returns:
            True if authentication successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def scrape(self, hours_back: int = 12, **kwargs) -> ScrapingResult:
        """
        Scrape data from the platform.
        
        Args:
            hours_back: Number of hours back to scrape data
            **kwargs: Additional platform-specific parameters
            
        Returns:
            ScrapingResult containing the scraped data and metadata
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (close connections, browsers, etc.)."""
        pass
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate a message against the standard format.
        
        Args:
            message: Message to validate
            
        Returns:
            True if valid, False otherwise
        """
        return validate_message(message)
    
    def format_message(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format raw platform data into the standard message format.
        
        Args:
            raw_data: Raw data from the platform
            
        Returns:
            Formatted message in standard format
        """
        # Extract common fields with safe defaults
        message_id = raw_data.get('id', '')
        author_name = raw_data.get('author_name', 'Unknown')
        author_id = raw_data.get('author_id', '')
        content_text = raw_data.get('text', '')
        timestamp = raw_data.get('timestamp')
        
        # Ensure timestamp is in ISO format
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp).isoformat()
        elif not timestamp:
            timestamp = datetime.now().isoformat()
        
        formatted_message = {
            'id': message_id,
            'platform': self.platform_name,
            'author': {
                'name': author_name,
                'id': author_id,
                'avatar_url': raw_data.get('avatar_url', '')
            },
            'content': {
                'text': content_text,
                'html': raw_data.get('html', ''),
                'media': raw_data.get('media_urls', [])
            },
            'metadata': {
                'posted_at': timestamp,
                'message_url': raw_data.get('url', ''),
                'reply_to': raw_data.get('reply_to', ''),
                'reactions': raw_data.get('reactions', {}),
                'scraped_at': datetime.now().isoformat()
            },
            'context': {
                'channel': raw_data.get('channel', ''),
                'server': raw_data.get('server', ''),
                'thread': raw_data.get('thread', ''),
                'group': raw_data.get('group', '')
            }
        }
        
        return formatted_message
    
    async def rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        
        # Check if we need to wait based on minimum delay
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        # Update request tracking
        self._last_request_time = asyncio.get_event_loop().time()
        self._request_count += 1
    
    async def retry_on_failure(self, operation, *args, **kwargs):
        """
        Retry an operation on failure with exponential backoff.
        
        Args:
            operation: Async function to retry
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception
    
    def get_time_range(self, hours_back: int) -> tuple[datetime, datetime]:
        """
        Get the time range for scraping.
        
        Args:
            hours_back: Number of hours back from now
            
        Returns:
            Tuple of (start_time, end_time) with local timezone
        """
        end_time = datetime.now()  # 使用本地时区
        start_time = end_time - timedelta(hours=hours_back)
        return start_time, end_time
    
    def is_message_in_time_range(self, message: Dict[str, Any], 
                                start_time: datetime, 
                                end_time: datetime) -> bool:
        """
        Check if a message falls within the specified time range.
        
        Args:
            message: Message to check
            start_time: Start of time range (local timezone)
            end_time: End of time range (local timezone)
            
        Returns:
            True if message is in range, False otherwise
        """
        try:
            posted_at_str = message.get('metadata', {}).get('posted_at', '')
            if not posted_at_str:
                return False
            
            # 智能解析时间戳到本地时区
            posted_at = self._parse_timestamp_to_local(posted_at_str)
            if not posted_at:
                return False
            
            return start_time <= posted_at <= end_time
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse timestamp: {e}")
            return False
    
    def _parse_timestamp_to_local(self, timestamp_str: str) -> Optional[datetime]:
        """
        智能解析时间戳到本地时区，兼容新旧数据格式。
        
        Args:
            timestamp_str: ISO格式时间戳字符串
            
        Returns:
            本地时区的datetime对象（naive，用于与time_range比较），失败返回None
        """
        try:
            # 解析ISO格式时间戳
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # 统一转换为本地时区的naive datetime用于比较
            if dt.tzinfo:
                # 有时区信息，转换到本地时区然后移除时区信息
                local_aware = dt.astimezone()
                return local_aware.replace(tzinfo=None)
            else:
                # 无时区信息，假设已经是本地时间
                return dt
                
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Failed to parse timestamp '{timestamp_str}': {e}")
            return None
    
    def filter_messages_by_time(self, messages: List[Dict[str, Any]], 
                               hours_back: int) -> List[Dict[str, Any]]:
        """
        Filter messages by time range.
        
        Args:
            messages: List of messages to filter
            hours_back: Number of hours back from now
            
        Returns:
            Filtered list of messages
        """
        start_time, end_time = self.get_time_range(hours_back)
        filtered_messages = []
        
        for message in messages:
            if self.is_message_in_time_range(message, start_time, end_time):
                filtered_messages.append(message)
        
        self.logger.info(
            f"Filtered {len(messages)} messages to {len(filtered_messages)} "
            f"within {hours_back} hours"
        )
        
        return filtered_messages
    
    def create_scraping_result(self, messages: List[Dict[str, Any]], 
                              errors: List[str] = None) -> ScrapingResult:
        """
        Create a ScrapingResult object.
        
        Args:
            messages: List of scraped messages
            errors: List of error messages
            
        Returns:
            ScrapingResult object
        """
        if errors is None:
            errors = []
        
        end_time = datetime.now()
        start_time = self._session_start_time or end_time
        
        # Count valid messages
        valid_messages = [msg for msg in messages if self.validate_message(msg)]
        
        return ScrapingResult(
            platform=self.platform_name,
            messages=valid_messages,
            total_count=len(messages),
            success_count=len(valid_messages),
            error_count=len(messages) - len(valid_messages) + len(errors),
            start_time=start_time,
            end_time=end_time,
            errors=errors
        )
    
    @log_async_function_call(TDXLogger.get_logger("tdxagent.scrapers.base"))
    async def scrape_with_monitoring(self, hours_back: int = 12, **kwargs) -> ScrapingResult:
        """
        Scrape data with comprehensive monitoring and error handling.
        
        Args:
            hours_back: Number of hours back to scrape
            **kwargs: Additional platform-specific parameters
            
        Returns:
            ScrapingResult with monitoring data
        """
        self._session_start_time = datetime.now()
        self.logger.info(f"Starting {self.platform_name} scraping session")
        
        try:
            # Authenticate if not already done
            if not self._is_authenticated:
                auth_success = await self.authenticate()
                if not auth_success:
                    raise Exception("Authentication failed")
                self._is_authenticated = True
            
            # Perform the actual scraping
            result = await self.scrape(hours_back, **kwargs)
            
            # Log results
            self.logger.info(
                f"Scraping completed: {result.success_count}/{result.total_count} "
                f"messages in {result.duration:.2f}s "
                f"({result.success_rate:.1f}% success rate)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Scraping failed: {e}")
            return self.create_scraping_result([], [str(e)])
        
        finally:
            # Always cleanup
            try:
                await self.cleanup()
            except Exception as e:
                self.logger.warning(f"Cleanup failed: {e}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(platform={self.platform_name})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"platform={self.platform_name}, "
                f"authenticated={self._is_authenticated})")
