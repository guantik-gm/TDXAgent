"""
Configuration management for TDXAgent.

This module provides a comprehensive configuration management system with:
- YAML file loading and validation
- Environment variable support
- Configuration hot-reloading
- Type validation and default values
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


@dataclass
class PlatformConfig:
    """Configuration for a specific platform."""
    enabled: bool = False
    
    
@dataclass
class TwitterConfig(PlatformConfig):
    """Twitter/X specific configuration."""
    headless: bool = False
    delay_range: list = field(default_factory=lambda: [2, 5])
    max_scrolls: int = 10
    cookie_file: str = "twitter_cookies.json"
    login_timeout: int = 600
    use_persistent_browser: bool = True
    user_data_dir: str = "twitter_user_data"
    
    # 新的收集策略配置
    collection_strategy: str = "count_based"  # count_based | time_based
    max_tweets_per_run: int = 100            # 每次运行最大推文数
    following_strategy: str = "fixed_count"   # fixed_count (后续支持: incremental)
    for_you_strategy: str = "fixed_count"     # fixed_count only
    
    # 增量更新配置（未来功能）
    incremental: dict = field(default_factory=lambda: {
        'enabled': False,
        'state_file': 'twitter_last_seen.json',
        'fallback_count': 100
    })
    
    # Twitter专用代理配置
    proxy: dict = field(default_factory=lambda: {
        'enabled': False,
        'type': 'socks5',
        'host': '127.0.0.1',
        'port': 7890
    })


@dataclass
class TelegramConfig(PlatformConfig):
    """Telegram specific configuration."""
    api_id: str = ""
    api_hash: str = ""
    session_name: str = "tdxagent_session"
    group_whitelist: list = field(default_factory=list)
    # 向后兼容的旧配置
    max_messages: int = 1000
    # 新的细粒度配置
    max_messages_per_group: int = 1000
    max_total_messages: int = 10000
    enable_per_group_limit: bool = True


@dataclass
class DiscordConfig(PlatformConfig):
    """Discord specific configuration."""
    mode: str = "safe"  # safe or experimental
    export_path: str = "discord_exports"
    experimental: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GmailConfig(PlatformConfig):
    """Gmail specific configuration."""
    credentials_file: str = "gmail_credentials.json"
    token_file: str = "gmail_token.json"
    filters: Dict[str, Any] = field(default_factory=lambda: {
        'labels': [],
        'from_addresses': [],
        'keywords': [],
        'exclude_spam': True
    })
    batch_size: int = 100
    max_results: int = 500
    min_delay: float = 0.5
    max_requests_per_minute: int = 120


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai"
    batch_size: int = 25
    max_tokens: int = 4000
    timeout: int = 30
    max_requests_per_minute: int = 30
    max_tokens_per_minute: int = 50000
    delay_between_batches: float = 2.0  # 统一多平台分析架构的批次间延迟
    max_retries: int = 3
    retry_delay: float = 2.0
    enable_prompt_files: bool = True
    openai: Dict[str, Any] = field(default_factory=dict)
    gemini: Dict[str, Any] = field(default_factory=dict)
    claude_cli: Dict[str, Any] = field(default_factory=dict)
    gemini_cli: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyConfig:
    """Proxy configuration."""
    enabled: bool = False
    type: str = "socks5"  # http, https, socks5
    host: str = "127.0.0.1"
    port: int = 7890
    username: str = ""
    password: str = ""


@dataclass
class AppConfig:
    """Main application configuration."""
    default_hours_to_fetch: int = 12
    data_directory: str = "TDXAgent_Data"
    log_level: str = "INFO"
    max_retries: int = 3
    max_concurrent_tasks: int = 3


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_modified = 0
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path == str(self.config_manager.config_path):
            # Debounce rapid file changes
            current_time = time.time()
            if current_time - self.last_modified > 1:
                self.last_modified = current_time
                self.config_manager._reload_config()


class ConfigManager:
    """
    Comprehensive configuration manager for TDXAgent.
    
    Features:
    - YAML configuration file loading
    - Environment variable override support
    - Configuration validation
    - Hot-reloading capability
    - Type-safe configuration access
    """
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self._config_data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._observer: Optional[Observer] = None
        
        # Configuration objects
        self.app: AppConfig = AppConfig()
        self.proxy: ProxyConfig = ProxyConfig()
        self.twitter: TwitterConfig = TwitterConfig()
        self.telegram: TelegramConfig = TelegramConfig()
        self.discord: DiscordConfig = DiscordConfig()
        self.gmail: GmailConfig = GmailConfig()
        self.llm: LLMConfig = LLMConfig()
        self.prompts: Dict[str, str] = {}
        self.output: Dict[str, Any] = {}
        
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from file and environment variables."""
        with self._lock:
            try:
                # Load from YAML file
                if self.config_path.exists():
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        self._config_data = yaml.safe_load(f) or {}
                    self.logger.info(f"Loaded configuration from {self.config_path}")
                else:
                    # Load default configuration
                    default_config_path = Path(__file__).parent / "default_config.yaml"
                    if default_config_path.exists():
                        with open(default_config_path, 'r', encoding='utf-8') as f:
                            self._config_data = yaml.safe_load(f) or {}
                        self.logger.warning(f"Config file not found, using defaults from {default_config_path}")
                    else:
                        self._config_data = {}
                        self.logger.warning("No configuration file found, using empty config")
                
                # Override with environment variables
                self._apply_env_overrides()
                
                # Parse and validate configuration
                self._parse_config()
                
                self.logger.info("Configuration loaded successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                raise
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'TDXAGENT_LOG_LEVEL': ['settings', 'log_level'],
            'TDXAGENT_DATA_DIR': ['settings', 'data_directory'],
            'OPENAI_API_KEY': ['llm', 'openai', 'api_key'],
            'OPENAI_BASE_URL': ['llm', 'openai', 'base_url'],
            'GEMINI_API_KEY': ['llm', 'gemini', 'api_key'],
            'TELEGRAM_API_ID': ['platforms', 'telegram', 'api_id'],
            'TELEGRAM_API_HASH': ['platforms', 'telegram', 'api_hash'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(self._config_data, config_path, value)
    
    def _set_nested_value(self, data: Dict[str, Any], path: list, value: Any) -> None:
        """Set a nested dictionary value using a path list."""
        for key in path[:-1]:
            data = data.setdefault(key, {})
        data[path[-1]] = value
    
    def _parse_config(self) -> None:
        """Parse and validate the loaded configuration data."""
        # Parse app settings
        settings = self._config_data.get('settings', {})
        self.app = AppConfig(
            default_hours_to_fetch=settings.get('default_hours_to_fetch', 12),
            data_directory=settings.get('data_directory', 'TDXAgent_Data'),
            log_level=settings.get('log_level', 'INFO'),
            max_retries=settings.get('max_retries', 3),
            max_concurrent_tasks=settings.get('max_concurrent_tasks', 3)
        )
        
        # Parse proxy settings
        proxy_config = self._config_data.get('proxy', {})
        self.proxy = ProxyConfig(
            enabled=proxy_config.get('enabled', False),
            type=proxy_config.get('type', 'socks5'),
            host=proxy_config.get('host', '127.0.0.1'),
            port=proxy_config.get('port', 7890),
            username=proxy_config.get('username', ''),
            password=proxy_config.get('password', '')
        )
        
        # Parse platform configurations
        platforms = self._config_data.get('platforms', {})
        
        # Twitter configuration
        twitter_config = platforms.get('twitter', {})
        self.twitter = TwitterConfig(
            enabled=twitter_config.get('enabled', False),
            headless=twitter_config.get('headless', False),
            delay_range=twitter_config.get('delay_range', [2, 5]),
            max_scrolls=twitter_config.get('max_scrolls', 10),
            cookie_file=twitter_config.get('cookie_file', 'twitter_cookies.json'),
            login_timeout=twitter_config.get('login_timeout', 600),
            use_persistent_browser=twitter_config.get('use_persistent_browser', True),
            user_data_dir=twitter_config.get('user_data_dir', 'twitter_user_data'),
            collection_strategy=twitter_config.get('collection_strategy', 'count_based'),
            max_tweets_per_run=twitter_config.get('max_tweets_per_run', 100),
            following_strategy=twitter_config.get('following_strategy', 'fixed_count'),
            for_you_strategy=twitter_config.get('for_you_strategy', 'fixed_count'),
            incremental=twitter_config.get('incremental', {
                'enabled': False,
                'state_file': 'twitter_last_seen.json',
                'fallback_count': 100
            }),
            proxy=twitter_config.get('proxy', {
                'enabled': False,
                'type': 'socks5',
                'host': '127.0.0.1',
                'port': 7890
            })
        )
        
        # Telegram configuration
        telegram_config = platforms.get('telegram', {})
        self.telegram = TelegramConfig(
            enabled=telegram_config.get('enabled', False),
            api_id=telegram_config.get('api_id', ''),
            api_hash=telegram_config.get('api_hash', ''),
            session_name=telegram_config.get('session_name', 'tdxagent_session'),
            group_whitelist=telegram_config.get('group_whitelist', []),
            # 向后兼容：如果没有新配置，使用旧的max_messages
            max_messages=telegram_config.get('max_messages', 1000),
            max_messages_per_group=telegram_config.get('max_messages_per_group', telegram_config.get('max_messages', 1000)),
            max_total_messages=telegram_config.get('max_total_messages', 10000),
            enable_per_group_limit=telegram_config.get('enable_per_group_limit', True)
        )
        
        # Discord configuration
        discord_config = platforms.get('discord', {})
        self.discord = DiscordConfig(
            enabled=discord_config.get('enabled', False),
            mode=discord_config.get('mode', 'safe'),
            export_path=discord_config.get('export_path', 'discord_exports'),
            experimental=discord_config.get('experimental', {})
        )
        
        # Gmail configuration
        gmail_config = platforms.get('gmail', {})
        self.gmail = GmailConfig(
            enabled=gmail_config.get('enabled', False),
            credentials_file=gmail_config.get('credentials_file', 'gmail_credentials.json'),
            token_file=gmail_config.get('token_file', 'gmail_token.json'),
            filters=gmail_config.get('filters', {
                'labels': [],
                'from_addresses': [],
                'keywords': [],
                'exclude_spam': True
            }),
            batch_size=gmail_config.get('batch_size', 100),
            max_results=gmail_config.get('max_results', 500),
            min_delay=gmail_config.get('min_delay', 0.5),
            max_requests_per_minute=gmail_config.get('max_requests_per_minute', 120)
        )
        
        # LLM configuration
        llm_config = self._config_data.get('llm', {})
        self.llm = LLMConfig(
            provider=llm_config.get('provider', 'openai'),
            batch_size=llm_config.get('batch_size', 25),
            max_tokens=llm_config.get('max_tokens', 4000),
            timeout=llm_config.get('timeout', 30),
            max_requests_per_minute=llm_config.get('max_requests_per_minute', 30),
            max_tokens_per_minute=llm_config.get('max_tokens_per_minute', 50000),
            delay_between_batches=llm_config.get('delay_between_batches', 2.0),
            max_retries=llm_config.get('max_retries', 3),
            retry_delay=llm_config.get('retry_delay', 2.0),
            enable_prompt_files=llm_config.get('enable_prompt_files', True),
            openai=llm_config.get('openai', {}),
            gemini=llm_config.get('gemini', {}),
            claude_cli=llm_config.get('claude_cli', {}),
            gemini_cli=llm_config.get('gemini_cli', {})
        )
        
        # Prompts and output configuration
        self.prompts = self._config_data.get('prompts', {})
        self.output = self._config_data.get('output', {})
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        errors = []
        
        # Validate LLM configuration
        if self.llm.provider == 'openai':
            if not self.llm.openai.get('api_key'):
                errors.append("OpenAI API key is required when using OpenAI provider")
        elif self.llm.provider == 'gemini':
            if not self.llm.gemini.get('api_key'):
                errors.append("Gemini API key is required when using Gemini provider")
        elif self.llm.provider == 'claude_cli':
            # Claude CLI doesn't require API key, but should validate CLI availability
            pass
        elif self.llm.provider == 'gemini_cli':
            # Gemini CLI doesn't require API key, but should validate CLI availability
            pass
        
        # Validate Telegram configuration
        if self.telegram.enabled:
            if not self.telegram.api_id or not self.telegram.api_hash:
                errors.append("Telegram API ID and hash are required when Telegram is enabled")
        
        # Validate Discord experimental mode
        if self.discord.enabled and self.discord.mode == 'experimental':
            if not self.discord.experimental.get('token'):
                errors.append("Discord token is required for experimental mode")
        
        if errors:
            for error in errors:
                self.logger.error(f"Configuration validation error: {error}")
            return False
        
        return True
    
    def enable_hot_reload(self) -> None:
        """Enable hot-reloading of configuration file."""
        if self._observer is not None:
            return
            
        try:
            self._observer = Observer()
            event_handler = ConfigFileHandler(self)
            self._observer.schedule(
                event_handler, 
                str(self.config_path.parent), 
                recursive=False
            )
            self._observer.start()
            self.logger.info("Configuration hot-reload enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable hot-reload: {e}")
    
    def disable_hot_reload(self) -> None:
        """Disable hot-reloading of configuration file."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self.logger.info("Configuration hot-reload disabled")
    
    def _reload_config(self) -> None:
        """Internal method to reload configuration."""
        try:
            self.logger.info("Reloading configuration...")
            self.load_config()
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
    
    def get_data_directory(self) -> Path:
        """Get the configured data directory as a Path object."""
        return Path(self.app.data_directory)
    
    def get_platform_config(self, platform: str) -> Optional[PlatformConfig]:
        """Get configuration for a specific platform."""
        platform_configs = {
            'twitter': self.twitter,
            'telegram': self.telegram,
            'discord': self.discord,
            'gmail': self.gmail
        }
        return platform_configs.get(platform.lower())
    
    def is_platform_enabled(self, platform: str) -> bool:
        """Check if a platform is enabled."""
        config = self.get_platform_config(platform)
        return config.enabled if config else False
    
    def get_prompt(self, platform: str) -> str:
        """Get the prompt template for a platform."""
        return self.prompts.get(platform, "")
    
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.disable_hot_reload()
