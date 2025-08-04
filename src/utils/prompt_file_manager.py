"""
Unified prompt file management system for TDXAgent.

This module provides centralized management of prompt files for all LLM providers,
enabling debugging, optimization, and consistent file handling.
"""

import os
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
import tempfile
import shutil

from utils.logger import TDXLogger
from utils.helpers import ensure_directory, sanitize_filename


class PromptFileManager:
    """
    Centralized prompt file management system.
    
    Features:
    - Unified file naming convention
    - Automatic sequence numbering
    - File cleanup management
    - Cross-provider compatibility
    - Debugging support
    """
    
    def __init__(self, base_directory: str = "TDXAgent_Data/prompts"):
        """
        Initialize prompt file manager.
        
        Args:
            base_directory: Base directory for prompt files
        """
        self.base_directory = Path(base_directory)
        self.logger = TDXLogger.get_logger("tdxagent.utils.prompt_files")
        
        # File naming configuration
        self.file_extension = ".txt"
        self.cleanup_days = 7  # Keep files for 7 days
        
        # Sequence tracking
        self._current_sequence = 0
        self._session_start = datetime.now()
        
        # Initialize directory
        self._initialize_directory()
        
        self.logger.info(f"Initialized prompt file manager: {self.base_directory}")
    
    def _initialize_directory(self) -> None:
        """Initialize the prompt directory."""
        try:
            ensure_directory(self.base_directory)
            
            # Find the current highest sequence number for today
            self._current_sequence = self._find_current_sequence()
            
            self.logger.debug(f"Starting sequence: {self._current_sequence}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize prompt directory: {e}")
            raise
    
    def _find_current_sequence(self) -> int:
        """Find the current highest sequence number for today."""
        today_prefix = self._session_start.strftime("%Y%m%d_%H%M%S")
        date_prefix = today_prefix[:8]  # YYYYMMDD
        
        max_sequence = 0
        
        try:
            for file_path in self.base_directory.glob(f"{date_prefix}_*{self.file_extension}"):
                filename = file_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    try:
                        sequence = int(parts[2])
                        max_sequence = max(max_sequence, sequence)
                    except ValueError:
                        continue
        except Exception as e:
            self.logger.warning(f"Error finding current sequence: {e}")
        
        return max_sequence
    
    def save_prompt(self, prompt: str, analysis_type: str = "analysis") -> str:
        """
        Save prompt to file with unified naming convention.
        
        Args:
            prompt: Prompt text to save
            analysis_type: Type of analysis (for debugging purposes)
            
        Returns:
            Path to saved prompt file
        """
        try:
            # Generate filename
            filename = self._generate_filename(analysis_type)
            file_path = self.base_directory / filename
            
            # Save prompt to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            self.logger.debug(f"Saved prompt: {filename} ({len(prompt)} chars)")
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save prompt: {e}")
            raise
    
    def save_prompt_temporary(self, prompt: str) -> str:
        """
        Save prompt to temporary file for immediate use.
        
        Args:
            prompt: Prompt text to save
            
        Returns:
            Path to temporary prompt file
        """
        try:
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(
                suffix=self.file_extension,
                prefix='tdx_prompt_',
                dir=str(self.base_directory)
            )
            
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            self.logger.debug(f"Saved temporary prompt: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to save temporary prompt: {e}")
            raise
    
    def load_prompt(self, file_path: str) -> str:
        """
        Load prompt from file.
        
        Args:
            file_path: Path to prompt file
            
        Returns:
            Prompt text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.debug(f"Loaded prompt: {file_path} ({len(content)} chars)")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to load prompt from {file_path}: {e}")
            raise
    
    def _generate_filename(self, analysis_type: str = "analysis") -> str:
        """
        Generate filename using unified naming convention.
        
        Args:
            analysis_type: Type of analysis (will be included in filename)
            
        Returns:
            Generated filename
        """
        # Increment sequence
        self._current_sequence += 1
        
        # Format: YYYYMMDD_HHMMSS_NNN_analysis_type.txt
        timestamp = self._session_start.strftime("%Y%m%d_%H%M%S")
        sequence = f"{self._current_sequence:03d}"
        
        # Sanitize analysis_type for filename use
        safe_analysis_type = analysis_type.replace(" ", "_").replace("/", "_")
        
        filename = f"{timestamp}_{sequence}_{safe_analysis_type}{self.file_extension}"
        
        return filename
    
    def cleanup_old_files(self, days: Optional[int] = None) -> int:
        """
        Clean up old prompt files.
        
        Args:
            days: Number of days to keep (default: self.cleanup_days)
            
        Returns:
            Number of files cleaned up
        """
        if days is None:
            days = self.cleanup_days
        
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        try:
            for file_path in self.base_directory.glob(f"*{self.file_extension}"):
                try:
                    # Check file modification time
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        self.logger.debug(f"Cleaned up old prompt file: {file_path.name}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old prompt files (older than {days} days)")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files: {e}")
            return 0
    
    def list_prompt_files(self, limit: Optional[int] = None) -> List[str]:
        """
        List prompt files in directory.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of prompt file paths (most recent first)
        """
        try:
            files = []
            
            for file_path in self.base_directory.glob(f"*{self.file_extension}"):
                if file_path.is_file():
                    files.append(str(file_path))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            if limit:
                files = files[:limit]
            
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list prompt files: {e}")
            return []
    
    def get_file_stats(self) -> dict:
        """
        Get statistics about prompt files.
        
        Returns:
            Dictionary with file statistics
        """
        try:
            files = list(self.base_directory.glob(f"*{self.file_extension}"))
            
            if not files:
                return {
                    'total_files': 0,
                    'total_size_bytes': 0,
                    'oldest_file': None,
                    'newest_file': None
                }
            
            total_size = sum(f.stat().st_size for f in files)
            file_times = [f.stat().st_mtime for f in files]
            
            return {
                'total_files': len(files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'oldest_file': datetime.fromtimestamp(min(file_times)),
                'newest_file': datetime.fromtimestamp(max(file_times)),
                'current_sequence': self._current_sequence
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get file stats: {e}")
            return {}
    
    def create_backup(self, backup_dir: str) -> bool:
        """
        Create backup of all prompt files.
        
        Args:
            backup_dir: Directory to create backup in
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = backup_path / f"prompts_backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            copied_count = 0
            
            for file_path in self.base_directory.glob(f"*{self.file_extension}"):
                if file_path.is_file():
                    dest_path = backup_subdir / file_path.name
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
            
            self.logger.info(f"Created backup of {copied_count} prompt files: {backup_subdir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def get_prompt_for_debugging(self, sequence: int) -> Optional[str]:
        """
        Get specific prompt file for debugging purposes.
        
        Args:
            sequence: Sequence number to find
            
        Returns:
            Prompt content if found, None otherwise
        """
        try:
            # Search for file with matching sequence
            for file_path in self.base_directory.glob(f"*_{sequence:03d}{self.file_extension}"):
                return self.load_prompt(str(file_path))
            
            self.logger.warning(f"No prompt file found for sequence {sequence}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get prompt for debugging: {e}")
            return None
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_file_stats()
        return f"PromptFileManager(dir={self.base_directory}, files={stats.get('total_files', 0)}, seq={self._current_sequence})"