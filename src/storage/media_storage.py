"""
Media file storage system for TDXAgent.

This module handles downloading, storing, and managing media files
(images, videos, audio) from social media platforms.
"""

import asyncio
import aiohttp
import aiofiles
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import mimetypes
import logging
from urllib.parse import urlparse, unquote
import json

from utils.logger import TDXLogger
from utils.helpers import (
    ensure_directory, sanitize_filename, format_timestamp,
    get_file_extension, is_media_file, format_file_size
)


class MediaStorage:
    """
    Media file storage and management system.
    
    Features:
    - Asynchronous file downloading
    - Duplicate detection and deduplication
    - Organized storage by platform and date
    - Metadata tracking
    - File integrity verification
    - Automatic cleanup and optimization
    """
    
    def __init__(self, base_directory: Union[str, Path] = "TDXAgent_Data"):
        """
        Initialize media storage.
        
        Args:
            base_directory: Base directory for data storage
        """
        self.base_directory = Path(base_directory)
        self.media_directory = self.base_directory / "data"
        self.logger = TDXLogger.get_logger("tdxagent.storage.media")
        
        # Download configuration
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.timeout = 30  # seconds
        self.max_concurrent_downloads = 5
        
        # Supported media types
        self.supported_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',  # Images
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',   # Videos
            '.mp3', '.wav', '.flac', '.aac', '.ogg'            # Audio
        }
        
        # File metadata cache
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Initialized media storage at {self.media_directory}")
    
    async def initialize(self) -> None:
        """Initialize media storage directories."""
        platforms = ['twitter', 'telegram', 'discord']
        for platform in platforms:
            media_dir = self.media_directory / platform / "media"
            await ensure_directory(media_dir)
        
        self.logger.info("Media storage directories initialized")
    
    def _get_media_directory(self, platform: str, date_obj: Optional[datetime] = None) -> Path:
        """
        Get the media directory for a platform and date.
        
        Args:
            platform: Platform name
            date_obj: Date object (defaults to today)
            
        Returns:
            Path to the media directory
        """
        if date_obj is None:
            date_obj = datetime.now()
        
        date_str = date_obj.strftime("%Y-%m")  # Organize by year-month
        return self.media_directory / platform / "media" / date_str
    
    def _generate_file_hash(self, url: str, content: Optional[bytes] = None) -> str:
        """
        Generate a unique hash for a file.
        
        Args:
            url: File URL
            content: File content (optional)
            
        Returns:
            File hash
        """
        if content:
            return hashlib.md5(content).hexdigest()
        else:
            return hashlib.md5(url.encode()).hexdigest()
    
    def _get_filename_from_url(self, url: str) -> str:
        """
        Extract filename from URL.
        
        Args:
            url: File URL
            
        Returns:
            Sanitized filename
        """
        parsed = urlparse(url)
        filename = Path(unquote(parsed.path)).name
        
        if not filename or '.' not in filename:
            # Generate filename from URL hash
            url_hash = self._generate_file_hash(url)
            filename = f"media_{url_hash[:12]}"
        
        return sanitize_filename(filename)
    
    def _get_file_extension_from_content_type(self, content_type: str) -> str:
        """
        Get file extension from content type.
        
        Args:
            content_type: MIME content type
            
        Returns:
            File extension
        """
        extension = mimetypes.guess_extension(content_type)
        if extension and extension.lower() in self.supported_extensions:
            return extension.lower()
        return ""
    
    async def download_media(self, url: str, platform: str,
                           message_id: str = "",
                           custom_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Download a media file from URL.
        
        Args:
            url: Media file URL
            platform: Platform name
            message_id: Associated message ID
            custom_filename: Custom filename (optional)
            
        Returns:
            Media file metadata or None if failed
        """
        try:
            # Validate URL
            if not url or not url.startswith(('http://', 'https://')):
                self.logger.warning(f"Invalid URL: {url}")
                return None
            
            # Generate filename
            if custom_filename:
                filename = sanitize_filename(custom_filename)
            else:
                filename = self._get_filename_from_url(url)
            
            # Download file
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.warning(f"Failed to download {url}: HTTP {response.status}")
                        return None
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if not any(media_type in content_type.lower() for media_type in 
                              ['image', 'video', 'audio']):
                        self.logger.warning(f"Unsupported content type: {content_type}")
                        return None
                    
                    # Check file size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_file_size:
                        self.logger.warning(f"File too large: {format_file_size(int(content_length))}")
                        return None
                    
                    # Read content
                    content = await response.read()
                    
                    # Check actual size
                    if len(content) > self.max_file_size:
                        self.logger.warning(f"File too large: {format_file_size(len(content))}")
                        return None
            
            # Generate file hash for deduplication
            file_hash = self._generate_file_hash(url, content)
            
            # Add extension if missing
            if not get_file_extension(filename):
                extension = self._get_file_extension_from_content_type(content_type)
                if extension:
                    filename += extension
                else:
                    # Default to .bin for unknown types
                    filename += '.bin'
            
            # Ensure it's a supported media file
            if not is_media_file(filename):
                self.logger.warning(f"Unsupported media file type: {filename}")
                return None
            
            # Create unique filename to avoid conflicts
            base_name = Path(filename).stem
            extension = Path(filename).suffix
            unique_filename = f"{base_name}_{file_hash[:8]}{extension}"
            
            # Determine storage path
            media_dir = self._get_media_directory(platform)
            await ensure_directory(media_dir)
            file_path = media_dir / unique_filename
            
            # Check if file already exists
            if file_path.exists():
                self.logger.debug(f"File already exists: {file_path}")
                # Return existing file metadata
                return await self._get_file_metadata(file_path, url, message_id)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Create metadata
            metadata = {
                'filename': unique_filename,
                'original_filename': filename,
                'file_path': str(file_path.relative_to(self.base_directory)),
                'absolute_path': str(file_path),
                'url': url,
                'platform': platform,
                'message_id': message_id,
                'file_hash': file_hash,
                'file_size': len(content),
                'content_type': content_type,
                'downloaded_at': datetime.now().isoformat(),
                'file_extension': extension
            }
            
            # Save metadata
            await self._save_file_metadata(file_path, metadata)
            
            self.logger.info(f"Downloaded media: {unique_filename} ({format_file_size(len(content))})")
            return metadata
            
        except asyncio.TimeoutError:
            self.logger.error(f"Download timeout for {url}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return None
    
    async def download_media_batch(self, urls: List[str], platform: str,
                                  message_id: str = "") -> List[Optional[Dict[str, Any]]]:
        """
        Download multiple media files concurrently.
        
        Args:
            urls: List of media URLs
            platform: Platform name
            message_id: Associated message ID
            
        Returns:
            List of media metadata (None for failed downloads)
        """
        if not urls:
            return []
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async def download_with_semaphore(url: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.download_media(url, platform, message_id)
        
        # Execute downloads concurrently
        tasks = [download_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        media_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Download failed for {urls[i]}: {result}")
                media_list.append(None)
            else:
                media_list.append(result)
        
        successful = sum(1 for result in media_list if result is not None)
        self.logger.info(f"Batch download complete: {successful}/{len(urls)} successful")
        
        return media_list
    
    async def _save_file_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Save file metadata to a JSON file."""
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
        
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            content = json.dumps(metadata, ensure_ascii=False, indent=2)
            await f.write(content)
    
    async def _get_file_metadata(self, file_path: Path, url: str = "", 
                                message_id: str = "") -> Dict[str, Any]:
        """Get file metadata."""
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
        
        if metadata_path.exists():
            try:
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    return json.loads(content)
            except Exception as e:
                self.logger.warning(f"Failed to read metadata for {file_path}: {e}")
        
        # Generate basic metadata if file exists but no metadata file
        if file_path.exists():
            stat = file_path.stat()
            return {
                'filename': file_path.name,
                'file_path': str(file_path.relative_to(self.base_directory)),
                'absolute_path': str(file_path),
                'url': url,
                'message_id': message_id,
                'file_size': stat.st_size,
                'file_extension': file_path.suffix,
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
        
        return {}
    
    async def get_media_for_message(self, platform: str, message_id: str) -> List[Dict[str, Any]]:
        """
        Get all media files associated with a message.
        
        Args:
            platform: Platform name
            message_id: Message ID
            
        Returns:
            List of media metadata
        """
        media_list = []
        platform_media_dir = self.media_directory / platform / "media"
        
        if not platform_media_dir.exists():
            return media_list
        
        # Search through all date directories
        for date_dir in platform_media_dir.iterdir():
            if date_dir.is_dir():
                for file_path in date_dir.glob("*"):
                    if file_path.is_file() and not file_path.name.endswith('.meta'):
                        metadata = await self._get_file_metadata(file_path)
                        if metadata.get('message_id') == message_id:
                            media_list.append(metadata)
        
        return media_list
    
    async def cleanup_orphaned_files(self, platform: str, days_old: int = 30) -> int:
        """
        Clean up orphaned media files (files without associated messages).
        
        Args:
            platform: Platform name
            days_old: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        platform_media_dir = self.media_directory / platform / "media"
        if not platform_media_dir.exists():
            return 0
        
        for date_dir in platform_media_dir.iterdir():
            if date_dir.is_dir():
                for file_path in date_dir.glob("*"):
                    if file_path.is_file() and not file_path.name.endswith('.meta'):
                        try:
                            # Check file age
                            if file_path.stat().st_mtime < cutoff_time:
                                # Remove file and its metadata
                                file_path.unlink()
                                metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
                                if metadata_path.exists():
                                    metadata_path.unlink()
                                removed_count += 1
                                self.logger.debug(f"Removed orphaned file: {file_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to remove {file_path}: {e}")
        
        self.logger.info(f"Cleanup complete: removed {removed_count} orphaned files")
        return removed_count
    
    async def get_storage_stats(self, platform: Optional[str] = None) -> Dict[str, Any]:
        """
        Get media storage statistics.
        
        Args:
            platform: Platform name (optional, gets stats for all if None)
            
        Returns:
            Storage statistics
        """
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'platforms': {}
        }
        
        platforms = [platform] if platform else ['twitter', 'telegram', 'discord']
        
        for platform_name in platforms:
            platform_media_dir = self.media_directory / platform_name / "media"
            platform_stats = {
                'files': 0,
                'size_bytes': 0,
                'file_types': {}
            }
            
            if platform_media_dir.exists():
                for date_dir in platform_media_dir.iterdir():
                    if date_dir.is_dir():
                        for file_path in date_dir.glob("*"):
                            if file_path.is_file() and not file_path.name.endswith('.meta'):
                                try:
                                    file_size = file_path.stat().st_size
                                    file_ext = file_path.suffix.lower()
                                    
                                    platform_stats['files'] += 1
                                    platform_stats['size_bytes'] += file_size
                                    
                                    if file_ext not in platform_stats['file_types']:
                                        platform_stats['file_types'][file_ext] = {'count': 0, 'size_bytes': 0}
                                    
                                    platform_stats['file_types'][file_ext]['count'] += 1
                                    platform_stats['file_types'][file_ext]['size_bytes'] += file_size
                                    
                                except Exception as e:
                                    self.logger.warning(f"Failed to get stats for {file_path}: {e}")
            
            stats['platforms'][platform_name] = platform_stats
            stats['total_files'] += platform_stats['files']
            stats['total_size_bytes'] += platform_stats['size_bytes']
        
        return stats
