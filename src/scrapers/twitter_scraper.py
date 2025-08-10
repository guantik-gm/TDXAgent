"""
Twitter/X scraper for TDXAgent using Playwright.

This module provides Twitter data collection using Playwright with advanced
anti-detection measures and human-like behavior simulation.
"""

import asyncio
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import logging

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from scrapers.base_scraper import BaseScraper, ScrapingResult
from utils.logger import TDXLogger, log_async_function_call
from storage.optimized_message_converter import OptimizedMessageConverter
from utils.helpers import ensure_directory, sanitize_filename
from utils.retry_handler import retry_on_failure, NetworkError, RateLimitError


class TwitterScraper(BaseScraper):
    """
    Twitter/X scraper using Playwright with anti-detection.
    
    Features:
    - Advanced anti-detection measures
    - Human-like behavior simulation
    - Cookie-based authentication
    - Following and For You timeline scraping
    - Media content extraction
    - Rate limiting compliance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Twitter scraper.
        
        Args:
            config: Twitter configuration dictionary
        """
        super().__init__(config, "twitter")
        
        # Browser configuration
        self.headless = config.get('headless', False)
        self.delay_range = config.get('delay_range', [2, 5])
        # 滚动次数限制已移除，但保留变量避免错误（设为极大值）
        self.max_scrolls = config.get('max_scrolls', 999999)
        self.cookie_file = config.get('cookie_file', 'twitter_cookies.json')
        
        # 滚动性能配置
        scroll_settings = config.get('scroll_settings', {})
        self.scroll_distance_range = scroll_settings.get('distance_range', [1500, 2500])
        self.content_wait_range = scroll_settings.get('content_wait_range', [2, 4])
        self.bottom_wait_range = scroll_settings.get('bottom_wait_range', [3, 6])
        self.scroll_mode = scroll_settings.get('mode', 'balanced')
        
        # 根据模式调整参数
        self._adjust_scroll_params_by_mode()
        self.login_timeout = config.get('login_timeout', 600)  # 默认10分钟超时
        
        # 持久化用户数据目录
        self.user_data_dir = config.get('user_data_dir', 'twitter_user_data')
        self.use_persistent_browser = config.get('use_persistent_browser', True)
        
        # Browser instances
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # 优化版转换器
        self.converter = OptimizedMessageConverter()
        
        # Anti-detection configuration
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # 调试信息：打印配置内容
        self.logger.info(f"Twitter scraper initialized with collection_strategy: {config.get('collection_strategy', 'NOT_FOUND')}")
        self.logger.debug(f"Available config keys: {list(config.keys())}")
        self.logger.info("Initialized Twitter scraper")
    
    @retry_on_failure(max_retries=3, base_delay=2.0, max_delay=30.0)
    async def authenticate(self) -> bool:
        """
        Authenticate with Twitter using cookies or interactive login.
        
        Returns:
            True if authentication successful
        """
        try:
            # Initialize Playwright
            self.logger.info("Starting Playwright...")
            self.playwright = await async_playwright().start()
            self.logger.info("Playwright started successfully")
            
            # Get proxy configuration from config
            from config.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            # Prepare browser launch args with SSL fixes
            launch_args = [
                '--no-blink-features=AutomationControlled',
                '--disable-blink-features=AutomationControlled', 
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--ignore-ssl-errors',
                '--ignore-certificate-errors',
                '--ignore-certificate-errors-spki-list',
                '--disable-ssl-false-start',
                '--disable-tls13-early-data',
                '--disable-features=VizDisplayCompositor'
            ]
            
            # Get proxy configuration (will be handled differently for persistent vs non-persistent)
            twitter_proxy = self.config.get('proxy', {})
            
            if self.use_persistent_browser:
                # 使用持久化浏览器上下文
                from pathlib import Path
                user_data_path = Path(self.user_data_dir)
                user_data_path.mkdir(exist_ok=True)
                
                self.logger.info(f"Launching persistent browser context with user data: {user_data_path}")
                
                # 使用 launch_persistent_context 而不是 launch + user-data-dir
                # 构建 proxy 参数
                proxy_config = None
                if twitter_proxy.get('enabled', False):
                    proxy_config = {
                        "server": f"{twitter_proxy.get('type', 'socks5')}://{twitter_proxy.get('host', '127.0.0.1')}:{twitter_proxy.get('port', 7890)}"
                    }
                    if twitter_proxy.get('username'):
                        proxy_config["username"] = twitter_proxy['username']
                        proxy_config["password"] = twitter_proxy['password']
                    self.logger.info(f"Using Twitter-specific proxy: {proxy_config['server']}")
                elif config_manager.proxy.enabled:
                    proxy_config = {
                        "server": f"{config_manager.proxy.type}://{config_manager.proxy.host}:{config_manager.proxy.port}"
                    }
                    if config_manager.proxy.username:
                        proxy_config["username"] = config_manager.proxy.username
                        proxy_config["password"] = config_manager.proxy.password
                    self.logger.info(f"Using global proxy: {proxy_config['server']}")
                else:
                    self.logger.info("No proxy configured for Twitter")
                
                self.context = await self.playwright.chromium.launch_persistent_context(
                    user_data_dir=str(user_data_path),
                    headless=self.headless,
                    args=launch_args,
                    proxy=proxy_config,
                    user_agent=random.choice(self.user_agents),
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                    timezone_id='America/New_York'
                )
                self.logger.info("Persistent browser context created successfully")
                
                # 获取已存在的页面或创建新页面
                if self.context.pages:
                    self.page = self.context.pages[0]
                    self.logger.info("Using existing page from persistent context")
                else:
                    self.page = await self.context.new_page()
                    self.logger.info("Created new page in persistent context")
                
            else:
                # 传统无痕模式
                # Add proxy args for non-persistent mode
                if twitter_proxy.get('enabled', False):
                    proxy_url = f"{twitter_proxy.get('type', 'socks5')}://"
                    if twitter_proxy.get('username'):
                        proxy_url += f"{twitter_proxy['username']}:{twitter_proxy['password']}@"
                    proxy_url += f"{twitter_proxy.get('host', '127.0.0.1')}:{twitter_proxy.get('port', 7890)}"
                    launch_args.append(f'--proxy-server={proxy_url}')
                    self.logger.info(f"Using Twitter-specific proxy: {twitter_proxy.get('type')}://{twitter_proxy.get('host')}:{twitter_proxy.get('port')}")
                elif config_manager.proxy.enabled:
                    proxy_url = f"{config_manager.proxy.type}://"
                    if config_manager.proxy.username:
                        proxy_url += f"{config_manager.proxy.username}:{config_manager.proxy.password}@"
                    proxy_url += f"{config_manager.proxy.host}:{config_manager.proxy.port}"
                    launch_args.append(f'--proxy-server={proxy_url}')
                    self.logger.info(f"Using global proxy for Twitter: {config_manager.proxy.type}://{config_manager.proxy.host}:{config_manager.proxy.port}")
                else:
                    self.logger.info("No proxy configured for Twitter")
                
                self.logger.info(f"Launching browser (headless={self.headless})...")
                self.browser = await self.playwright.chromium.launch(
                    headless=self.headless,
                    args=launch_args
                )
                self.logger.info("Browser launched successfully")
                
                # Create context with anti-detection
                self.logger.info("Creating browser context...")
                self.context = await self.browser.new_context(
                    user_agent=random.choice(self.user_agents),
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                    timezone_id='America/New_York',
                    permissions=['geolocation'],
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Cache-Control': 'max-age=0'
                    }
                )
                
                # Create page for non-persistent mode
                self.page = await self.context.new_page()
                
            self.logger.info("Browser context ready")
            
            # Page should already be created in both modes above
            if not self.page:
                self.logger.info("Creating new page...")
                self.page = await self.context.new_page()
                self.logger.info("Page created successfully")
            
            # Add stealth scripts
            self.logger.info("Adding stealth scripts...")
            await self._add_stealth_scripts()
            self.logger.info("Stealth scripts added")
            
            # Load cookies after page creation (仅在非持久化模式下)
            if not self.use_persistent_browser:
                self.logger.info("Loading cookies from file...")
                await self._load_cookies()
            else:
                self.logger.info("Using persistent browser - cookies automatically loaded")
            
            # Navigate to Twitter with shorter timeout
            self.logger.info("Navigating to Twitter homepage...")
            try:
                await self.page.goto('https://x.com/home', wait_until='domcontentloaded', timeout=30000)  # 30秒超时，使用domcontentloaded
                self.logger.info("Successfully navigated to Twitter")
                
                # Wait for basic content to load
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.warning(f"Navigation issue: {e}, trying to continue...")
                # Try simpler navigation
                try:
                    await self.page.goto('https://x.com/home', timeout=15000)
                    await asyncio.sleep(3)
                    self.logger.info("Fallback navigation successful")
                except Exception as e2:
                    self.logger.error(f"Fallback navigation failed: {e2}")
            
            # Check if logged in
            self.logger.info("Checking login status...")
            await asyncio.sleep(3)
            
            if await self._is_logged_in():
                self.logger.info("Already authenticated with existing cookies")
                self._is_authenticated = True
                return True
            else:
                self.logger.info("Not logged in, manual authentication required")
                return await self._interactive_login()
                
        except Exception as e:
            self.logger.error(f"Twitter authentication failed: {e}")
            return False
    
    async def _load_cookies(self) -> None:
        """Load cookies from file."""
        cookie_path = Path(self.cookie_file)
        
        if cookie_path.exists():
            try:
                self.logger.info(f"Found cookie file: {cookie_path}")
                with open(cookie_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.logger.info(f"Cookie file size: {len(content)} characters")
                    
                # Parse JSON with better error handling
                try:
                    cookies = json.loads(content)
                    self.logger.info(f"Parsed {len(cookies)} cookies")
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error at position {e.pos}: {e.msg}")
                    self.logger.error(f"Content around error: '{content[max(0, e.pos-20):e.pos+20]}'")
                    return
                
                # Validate cookie format
                for i, cookie in enumerate(cookies):
                    if not isinstance(cookie, dict) or 'name' not in cookie or 'value' not in cookie:
                        self.logger.error(f"Invalid cookie format at index {i}: {cookie}")
                        return
                
                await self.context.add_cookies(cookies)
                self.logger.info(f"Successfully loaded {len(cookies)} cookies")
                
                # Debug: Show which cookies were loaded
                cookie_names = [c['name'] for c in cookies]
                self.logger.info(f"Loaded cookies: {cookie_names}")
                
                # Check for important login cookies
                important_cookies = ['auth_token', 'ct0', 'twid']
                missing_cookies = [c for c in important_cookies if c not in cookie_names]
                if missing_cookies:
                    self.logger.warning(f"Missing important login cookies: {missing_cookies}")
                else:
                    self.logger.info("All important login cookies present")
                
            except Exception as e:
                self.logger.warning(f"Failed to load cookies: {e}")
        else:
            self.logger.info(f"No cookie file found at: {cookie_path}")
    
    async def _save_cookies(self) -> None:
        """Save cookies to file."""
        try:
            cookies = await self.context.cookies()
            
            with open(self.cookie_file, 'w') as f:
                json.dump(cookies, f, indent=2)
            
            self.logger.info("Saved cookies")
            
        except Exception as e:
            self.logger.warning(f"Failed to save cookies: {e}")
    
    async def _add_stealth_scripts(self) -> None:
        """Add stealth scripts to avoid detection."""
        stealth_script = """
        // Remove webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Mock languages and plugins
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        // Mock permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        """
        
        await self.page.add_init_script(stealth_script)
    
    async def _is_logged_in(self) -> bool:
        """Check if user is logged in."""
        try:
            self.logger.info("Checking login status...")
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Check current URL first
            current_url = self.page.url
            self.logger.info(f"Current URL: {current_url}")
            
            # If we're on login page, definitely not logged in
            if '/login' in current_url or '/i/flow/login' in current_url:
                self.logger.info("On login page - not logged in")
                return False
            
            # Look for login indicators (these mean NOT logged in)
            login_selectors = [
                '[data-testid="loginButton"]',
                '[href="/login"]',
                'text=Log in',
                'text=Sign up'
            ]
            
            for selector in login_selectors:
                count = await self.page.locator(selector).count()
                if count > 0:
                    self.logger.info(f"Found login indicator: {selector}")
                    return False
            
            # Look for logged-in indicators
            logged_in_selectors = [
                '[data-testid="SideNav_AccountSwitcher_Button"]',
                '[data-testid="AppTabBar_Profile_Link"]',
                '[aria-label="Profile"]',
                '[data-testid="primaryColumn"]',
                '[data-testid="tweet"]'
            ]
            
            for selector in logged_in_selectors:
                count = await self.page.locator(selector).count()
                if count > 0:
                    self.logger.info(f"Found logged-in indicator: {selector}")
                    return True
            
            # Final check: if we're on home page and no login buttons, assume logged in
            if '/home' in current_url:
                self.logger.info("On home page with no login indicators - assuming logged in")
                return True
            
            self.logger.info("No clear indicators found - not logged in")
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to check login status: {e}")
            return False
    
    async def _interactive_login(self) -> bool:
        """Handle interactive login process."""
        try:
            self.logger.info("Please log in manually in the browser window")
            
            if self.headless:
                self.logger.error("Cannot perform interactive login in headless mode")
                return False
            
            # Wait for user to log in
            print("\n" + "="*50)
            print("请在浏览器窗口中手动登录 Twitter/X")
            print("登录完成后，请按 Enter 键继续...")
            print("或者等待 30 秒自动检查登录状态...")
            print("="*50)
            
            try:
                # 使用 timeout 处理用户输入
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 30) == ([sys.stdin], [], []):
                    input()  # 用户按了Enter
                else:
                    self.logger.info("30秒超时，自动检查登录状态...")
            except:
                # 如果select不可用（Windows），使用简单的input
                input()
            
            # Check if now logged in
            if await self._is_logged_in():
                # 仅在非持久化模式下保存Cookie到文件
                if not self.use_persistent_browser:
                    await self._save_cookies()
                else:
                    self.logger.info("Persistent browser - cookies automatically saved")
                self._is_authenticated = True
                self.logger.info("Interactive login successful")
                return True
            else:
                self.logger.error("Login verification failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Interactive login failed: {e}")
            return False
    
    async def scrape(self, hours_back: int = 12, **kwargs) -> ScrapingResult:
        """
        Scrape Twitter data from Following and For You timelines.
        
        Args:
            hours_back: Number of hours back to scrape
            **kwargs: Additional parameters
            
        Returns:
            ScrapingResult with collected tweets
        """
        if not self.page or not self._is_authenticated:
            error_msg = "Twitter scraper not authenticated"
            self.logger.error(error_msg)
            return self.create_scraping_result([], [error_msg])
        
        messages = []
        errors = []
        
        try:
            # Scrape Following timeline
            self.logger.info("Scraping Following timeline...")
            following_tweets = await self._scrape_timeline('following', hours_back)
            messages.extend(following_tweets)
            
            # Random delay between timelines
            await asyncio.sleep(random.uniform(3, 7))
            
            # Scrape For You timeline
            self.logger.info("Scraping For You timeline...")
            for_you_tweets = await self._scrape_timeline('for_you', hours_back)
            messages.extend(for_you_tweets)
            
            # Convert to unified format
            converted_messages = []
            for raw_tweet in messages:
                try:
                    converted = self.converter.convert_message(raw_tweet, 'twitter')
                    if converted:
                        converted_messages.append(converted)
                except Exception as e:
                    self.logger.warning(f"Failed to convert tweet: {e}")
            
            self.logger.info(f"Scraped {len(converted_messages)} tweets from Twitter")
            return self.create_scraping_result(converted_messages, errors)
            
        except Exception as e:
            error_msg = f"Twitter scraping failed: {e}"
            self.logger.error(error_msg)
            return self.create_scraping_result([], [error_msg])
    
    async def _scrape_timeline(self, timeline_type: str, hours_back: int) -> List[Dict[str, Any]]:
        """
        Scrape a specific timeline using count-based or time-based strategy.
        
        Args:
            timeline_type: 'following' or 'for_you'
            hours_back: Number of hours back to scrape (only used for time_based strategy)
            
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        
        try:
            # Navigate to appropriate timeline
            self.logger.info(f"Navigating to {timeline_type} timeline...")
            if timeline_type == 'following':
                navigation_success = await self._navigate_to_following_timeline()
                if not navigation_success:
                    self.logger.warning("Following timeline navigation failed, continuing with current page")
            else:  # for_you
                navigation_success = await self._navigate_to_for_you_timeline()
                if not navigation_success:
                    self.logger.warning("For You timeline navigation failed, continuing with current page")
            
            # 检查收集策略
            collection_strategy = self.config.get('collection_strategy', 'time_based')
            self.logger.info(f"使用收集策略: {collection_strategy}")
            self.logger.debug(f"Twitter配置内容: {list(self.config.keys())}")
            
            if collection_strategy == 'count_based':
                # 不同页面使用不同策略
                if timeline_type == 'following':
                    # Following页面：时间优先策略
                    self.logger.info(f"📅 Following页面使用时间优先策略（时间限制>{hours_back}h，数量限制作为备选）")
                    tweets = await self._scrape_timeline_time_first(timeline_type, hours_back)
                else:  # for_you
                    # For You页面：纯数量策略（忽略时间）
                    self.logger.info(f"🔢 For You页面使用纯数量策略（忽略时间限制，只收集指定数量）")
                    tweets = await self._scrape_timeline_by_count(timeline_type, hours_back=None)
            else:
                tweets = await self._scrape_timeline_by_time(timeline_type, hours_back)
                
        except Exception as e:
            self.logger.error(f"Error scraping {timeline_type} timeline: {e}")
            
        return tweets
    
    async def _scrape_timeline_by_count(self, timeline_type: str, hours_back: int = None) -> List[Dict[str, Any]]:
        """基于推文数量的收集策略（仍然遵守时间边界）"""
        tweets = []
        max_tweets = self.config.get('max_tweets_per_run', 100)
        
        # 计算时间边界（如果提供了hours_back参数）
        cutoff_time = None
        if hours_back is not None:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            self.logger.info(f"Starting count-based tweet collection for {timeline_type} timeline with time boundary...")
            self.logger.info(f"Target: {max_tweets} tweets (within {hours_back} hours from {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            self.logger.info(f"Starting count-based tweet collection for {timeline_type} timeline...")
            self.logger.info(f"Target: {max_tweets} tweets (no time boundary)")
        
        scroll_count = 0
        collected_tweet_ids = set()  # 用于去重
        no_new_tweets_count = 0  # 连续没有新推文的次数
        consecutive_old_tweets = 0  # 连续超时推文计数（仅在有时间边界时使用）
        max_consecutive_old = 10  # 连续超时推文限制
        
        while len(tweets) < max_tweets:
            self.logger.info(f"Scroll {scroll_count + 1} - Extracting tweets from current view...")
            
            # Extract tweets from current view
            current_tweets = await self._extract_tweets_from_page(timeline_type)
            
            # 去重处理和时间过滤
            new_tweets = []
            valid_tweets = []  # 在时间范围内的推文
            old_tweets_in_batch = 0  # 本批次中的超时推文数量
            oldest_tweet_time = None
            
            for tweet in current_tweets:
                tweet_id = tweet.get('id', f"tweet_{random.randint(1000000, 9999999)}")
                if tweet_id not in collected_tweet_ids:
                    collected_tweet_ids.add(tweet_id)
                    new_tweets.append(tweet)
                    
                    # 时间过滤（如果设置了时间边界）
                    if cutoff_time is not None:
                        try:
                            tweet_time_str = tweet.get('created_at', '')
                            is_retweet = tweet.get('is_retweet', False)
                            tweet_time = self._parse_tweet_timestamp(tweet_time_str)
                            
                            # cutoff_time 确保是 naive datetime（无时区信息）
                            if cutoff_time.tzinfo is not None:
                                cutoff_time = cutoff_time.replace(tzinfo=None)
                            
                            # 跟踪最旧的推文时间
                            if oldest_tweet_time is None or tweet_time < oldest_tweet_time:
                                oldest_tweet_time = tweet_time
                            
                            # 检查是否在时间范围内
                            if tweet_time >= cutoff_time:
                                if is_retweet:
                                    self.logger.debug(f"Count-based: Retweet {tweet_id} within time range (retweet time: {tweet_time})")
                                else:
                                    self.logger.debug(f"Count-based: Original tweet {tweet_id} within time range (created: {tweet_time})")
                                valid_tweets.append(tweet)
                            else:
                                old_tweets_in_batch += 1
                                if is_retweet:
                                    self.logger.debug(f"Count-based: Retweet {tweet_id} too old (retweet time: {tweet_time}), not collecting")
                                else:
                                    self.logger.debug(f"Count-based: Tweet {tweet_id} too old (created: {tweet_time}), not collecting")
                        except Exception as e:
                            # 如果无法解析时间，包含这条推文（保守策略）
                            self.logger.debug(f"Failed to parse tweet time: {e}, including tweet")
                            valid_tweets.append(tweet)
                    else:
                        # 没有时间边界，添加所有新推文
                        valid_tweets.append(tweet)
            
            # 添加有效推文到结果列表
            tweets.extend(valid_tweets)
            
            # 如果超出目标数量，截断到目标数
            if len(tweets) > max_tweets:
                tweets = tweets[:max_tweets]
            
            new_tweets_count = len(new_tweets)
            valid_tweets_count = len(valid_tweets)
            total_tweets = len(tweets)
            
            # 更新连续超时推文计数（仅在有时间边界时）
            if cutoff_time is not None:
                if old_tweets_in_batch > 0 and valid_tweets_count == 0:
                    consecutive_old_tweets += old_tweets_in_batch
                else:
                    consecutive_old_tweets = 0  # 重置计数
                
                # 显示时间信息
                if oldest_tweet_time:
                    time_info = f", oldest: {oldest_tweet_time.strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    time_info = ""
                
                self.logger.info(f"发现 {len(current_tweets)} 条推文, {new_tweets_count} 条新推文, {valid_tweets_count} 条有效推文, 总计: {total_tweets}/{max_tweets}{time_info}")
                
                if old_tweets_in_batch > 0:
                    self.logger.info(f"本批次 {old_tweets_in_batch} 条推文超出时间限制, 连续超时: {consecutive_old_tweets}/{max_consecutive_old}")
                
                # 检查是否连续太多超时推文
                if consecutive_old_tweets >= max_consecutive_old:
                    self.logger.info(f"⏰ 时间限制触发：连续发现 {consecutive_old_tweets} 条超出时间限制的推文")
                    self.logger.info(f"📊 当前状态：时间边界={cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}, 最旧推文时间={oldest_tweet_time.strftime('%Y-%m-%d %H:%M:%S') if oldest_tweet_time else 'None'}")
                    self.logger.info(f"🛑 停止收集 - 已达到连续超时推文限制({max_consecutive_old}条)")
                    break
            else:
                self.logger.info(f"发现 {len(current_tweets)} 条推文, {new_tweets_count} 条新推文, 总计: {total_tweets}/{max_tweets}")
            
            # 检查是否已达到目标数量
            if len(tweets) >= max_tweets:
                self.logger.info(f"已收集到目标数量 {max_tweets} 条推文，停止收集")
                break
            
            # 检查是否连续没有新推文
            if new_tweets_count == 0:
                no_new_tweets_count += 1
                self.logger.warning(f"本次滚动未获取到新推文 ({no_new_tweets_count}/3)")
                if no_new_tweets_count >= 3:
                    self.logger.info("连续3次滚动未获取到新推文，停止收集")
                    break
            else:
                no_new_tweets_count = 0  # 重置计数
            
            # 模拟人类滚动行为
            await self._human_like_scroll()
            scroll_count += 1
            
            # 随机延迟
            delay = random.uniform(*self.delay_range)
            await asyncio.sleep(delay)
            
            # 检查是否需要处理意外导航
            current_url = self.page.url
            if '/status/' in current_url and 'home' not in current_url:
                self.logger.warning("检测到意外导航到推文详情页")
                recovery_success = await self._recover_from_accidental_navigation('https://x.com/home')
                if not recovery_success:
                    self.logger.error("无法从意外导航中恢复，停止收集")
                    break
        
        if cutoff_time is not None:
            self.logger.info(f"基于数量的收集完成（带时间边界）：收集了 {len(tweets)} 条推文（{hours_back}小时内）")
        else:
            self.logger.info(f"基于数量的收集完成（纯数量策略）：收集了 {len(tweets)} 条推文（For You页面，忽略时间）")
        return tweets
    
    async def _scrape_timeline_time_first(self, timeline_type: str, hours_back: int) -> List[Dict[str, Any]]:
        """时间优先策略：专用于Following页面，时间限制优先于条数限制"""
        tweets = []
        max_tweets = self.config.get('max_tweets_per_run', 100)
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        self.logger.info(f"Starting time-first tweet collection for {timeline_type} timeline...")
        self.logger.info(f"时间限制优先: {hours_back}小时内的推文 (从 {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')} 开始)")
        self.logger.info(f"最大数量限制: {max_tweets} 条推文 (时间限制优先)")
        self.logger.info(f"当前本地时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        scroll_count = 0
        collected_tweet_ids = set()  # 用于去重
        no_new_tweets_count = 0  # 连续没有新推文的次数
        consecutive_old_tweets = 0  # 连续超时推文计数
        max_consecutive_old = 3  # Following页面时间敏感，连续3条就停止
        
        while len(tweets) < max_tweets:
            self.logger.info(f"Scroll {scroll_count + 1} - Extracting tweets from current view...")
            
            # Extract tweets from current view
            current_tweets = await self._extract_tweets_from_page(timeline_type)
            
            # 去重处理和严格的时间过滤
            new_tweets = []
            valid_tweets = []  # 在时间范围内的推文
            old_tweets_in_batch = 0  # 本批次中的超时推文数量
            oldest_tweet_time = None
            
            for tweet in current_tweets:
                tweet_id = tweet.get('id', f"tweet_{random.randint(1000000, 9999999)}")
                if tweet_id not in collected_tweet_ids:
                    collected_tweet_ids.add(tweet_id)
                    new_tweets.append(tweet)
                    
                    # 严格的时间过滤（Following页面时间优先策略）
                    try:
                        tweet_time_str = tweet.get('created_at', '')
                        is_retweet = tweet.get('is_retweet', False)
                        tweet_time = self._parse_tweet_timestamp(tweet_time_str)
                        
                        # cutoff_time 确保是 naive datetime（无时区信息）
                        if cutoff_time.tzinfo is not None:
                            cutoff_time = cutoff_time.replace(tzinfo=None)
                        
                        # 跟踪最旧的推文时间
                        if oldest_tweet_time is None or tweet_time < oldest_tweet_time:
                            oldest_tweet_time = tweet_time
                        
                        # 检查是否在时间范围内
                        if tweet_time >= cutoff_time:
                            if is_retweet:
                                self.logger.debug(f"Following: Retweet {tweet_id} within time range (retweet time: {tweet_time})")
                            else:
                                self.logger.debug(f"Following: Original tweet {tweet_id} within time range (created: {tweet_time})")
                            valid_tweets.append(tweet)
                        else:
                            old_tweets_in_batch += 1
                            if is_retweet:
                                self.logger.debug(f"Following: Retweet {tweet_id} too old (retweet time: {tweet_time}), not collecting")
                            else:
                                self.logger.debug(f"Following: Tweet {tweet_id} too old (created: {tweet_time}), not collecting")
                    except Exception as e:
                        # 如果无法解析时间，包含这条推文（保守策略）
                        self.logger.debug(f"Failed to parse tweet time: {e}, including tweet")
                        valid_tweets.append(tweet)
            
            # 添加有效推文到结果列表
            tweets.extend(valid_tweets)
            
            new_tweets_count = len(new_tweets)
            valid_tweets_count = len(valid_tweets)
            total_tweets = len(tweets)
            
            # 更新连续超时推文计数 - 时间优先策略更严格
            if old_tweets_in_batch > 0 and valid_tweets_count == 0:
                consecutive_old_tweets += old_tweets_in_batch
            else:
                consecutive_old_tweets = 0  # 重置计数
            
            # 显示时间信息
            if oldest_tweet_time:
                time_info = f", oldest: {oldest_tweet_time.strftime('%Y-%m-%d %H:%M:%S')}"
            else:
                time_info = ""
            
            self.logger.info(f"发现 {len(current_tweets)} 条推文, {new_tweets_count} 条新推文, {valid_tweets_count} 条有效推文, 总计: {total_tweets}/{max_tweets}{time_info}")
            
            if old_tweets_in_batch > 0:
                self.logger.info(f"本批次 {old_tweets_in_batch} 条推文超出时间限制, 连续超时: {consecutive_old_tweets}/{max_consecutive_old}")
            
            # 时间优先策略：连续少量超时推文就停止
            if consecutive_old_tweets >= max_consecutive_old:
                self.logger.info(f"🕒 时间限制触发：连续发现 {consecutive_old_tweets} 条超出 {hours_back} 小时时间限制的推文")
                self.logger.info(f"📊 Following页面状态：时间边界={cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}, 最旧推文={oldest_tweet_time.strftime('%Y-%m-%d %H:%M:%S') if oldest_tweet_time else 'None'}")
                self.logger.info(f"🛑 Following页面停止收集 - 时间限制优先策略(连续{max_consecutive_old}条超时)")
                break
            
            # 检查是否已达到数量限制（次要条件）
            if len(tweets) >= max_tweets:
                self.logger.info(f"📊 数量限制触发：已收集到目标数量 {max_tweets} 条推文，停止收集")
                break
            
            # 检查是否连续没有新推文
            if new_tweets_count == 0:
                no_new_tweets_count += 1
                self.logger.warning(f"本次滚动未获取到新推文 ({no_new_tweets_count}/3)")
                if no_new_tweets_count >= 3:
                    self.logger.info("连续3次滚动未获取到新推文，停止收集")
                    break
            else:
                no_new_tweets_count = 0  # 重置计数
            
            # 检查导航状态，防止意外跳转到推文详情页
            current_url = self.page.url
            if '/status/' in current_url and 'home' not in current_url:
                self.logger.warning("检测到意外导航到推文详情页")
                recovery_success = await self._recover_from_accidental_navigation('https://x.com/home')
                if not recovery_success:
                    self.logger.error("无法从意外导航中恢复，停止收集")
                    break
            
            # Human-like scrolling
            await self._human_like_scroll()
            scroll_count += 1
            
            # 随机延迟
            delay = random.uniform(*self.delay_range)
            await asyncio.sleep(delay)
        
        self.logger.info(f"时间优先策略收集完成：收集了 {len(tweets)} 条推文（{hours_back}小时内，Following页面）")
        return tweets
    
    async def _scrape_timeline_by_time(self, timeline_type: str, hours_back: int) -> List[Dict[str, Any]]:
        """基于时间的收集策略（原有逻辑）"""
        tweets = []
        
        # Scroll and collect tweets with intelligent time-based stopping
        self.logger.info(f"Starting time-based tweet collection for {timeline_type} timeline...")
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        self.logger.info(f"Collecting tweets from {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')} onwards")
        
        scroll_count = 0
        collected_tweet_ids = set()  # 用于去重
        no_new_tweets_count = 0  # 连续没有新推文的次数
        consecutive_old_tweets = 0  # 连续超时推文计数
        max_consecutive_old = 15 if timeline_type == 'for_you' else 8  # For You页面更宽松
        reached_time_limit = False
        
        while not reached_time_limit:
            self.logger.info(f"Scroll {scroll_count + 1} - Extracting tweets from current view...")
            
            # Extract tweets from current view
            current_tweets = await self._extract_tweets_from_page(timeline_type)
            
            # 去重处理和智能时间检查（支持转发推文时间判断）
            new_tweets = []
            valid_tweets = []  # 在时间范围内的推文
            old_tweets_in_batch = 0  # 本批次中的超时推文数量
            oldest_tweet_time = None
            
            for tweet in current_tweets:
                tweet_id = tweet.get('id', f"tweet_{random.randint(1000000, 9999999)}")
                if tweet_id not in collected_tweet_ids:
                    collected_tweet_ids.add(tweet_id)
                    new_tweets.append(tweet)
                    
                    # 检查推文时间（基于时间策略）
                    try:
                        tweet_time_str = tweet.get('created_at', '')
                        is_retweet = tweet.get('is_retweet', False)
                        tweet_time = self._parse_tweet_timestamp(tweet_time_str)
                        
                        # cutoff_time 确保是 naive datetime（无时区信息）
                        if cutoff_time.tzinfo is not None:
                            cutoff_time = cutoff_time.replace(tzinfo=None)
                        
                        # 跟踪最旧的推文时间
                        if oldest_tweet_time is None or tweet_time < oldest_tweet_time:
                            oldest_tweet_time = tweet_time
                        
                        # 检查是否在时间范围内
                        if tweet_time >= cutoff_time:
                            if is_retweet:
                                self.logger.debug(f"Time-based: Retweet {tweet_id} within time range (retweet time: {tweet_time})")
                            else:
                                self.logger.debug(f"Time-based: Original tweet {tweet_id} within time range (created: {tweet_time})")
                            valid_tweets.append(tweet)
                        else:
                            old_tweets_in_batch += 1
                            if is_retweet:
                                self.logger.debug(f"Time-based: Retweet {tweet_id} too old (retweet time: {tweet_time}), not collecting")
                            else:
                                self.logger.debug(f"Time-based: Tweet {tweet_id} too old (created: {tweet_time}), not collecting")
                    except Exception as e:
                        # 如果无法解析时间，包含这条推文（保守策略）
                        self.logger.debug(f"Failed to parse tweet time: {e}, including tweet")
                        valid_tweets.append(tweet)
            
            # 只添加时间范围内的推文到最终结果
            tweets.extend(valid_tweets)
            
            new_tweets_count = len(new_tweets)
            valid_tweets_count = len(valid_tweets)
            total_tweets = len(tweets)
            
            # 更新连续超时推文计数
            if old_tweets_in_batch > 0 and valid_tweets_count == 0:
                consecutive_old_tweets += old_tweets_in_batch
            else:
                consecutive_old_tweets = 0  # 重置计数
            
            # 显示详细信息
            if oldest_tweet_time:
                time_info = f", oldest: {oldest_tweet_time.strftime('%Y-%m-%d %H:%M:%S')}"
                
                # 智能停止判断
                if consecutive_old_tweets >= max_consecutive_old:
                    reached_time_limit = True
                    self.logger.info(f"⏰ 基于时间策略触发：连续发现 {consecutive_old_tweets} 条超出时间限制的推文")
                    self.logger.info(f"📊 时间策略状态：时间边界={cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}, 最旧推文={oldest_tweet_time.strftime('%Y-%m-%d %H:%M:%S') if oldest_tweet_time else 'None'}")
                    self.logger.info(f"🛑 基于时间策略停止收集 - 连续{max_consecutive_old}条超时推文")
            else:
                time_info = ""
            
            self.logger.info(f"发现 {len(current_tweets)} 条推文, {new_tweets_count} 条新推文, {valid_tweets_count} 条有效推文, 总计: {total_tweets}{time_info}")
            
            if old_tweets_in_batch > 0:
                self.logger.info(f"本批次 {old_tweets_in_batch} 条推文超出时间限制, 连续超时: {consecutive_old_tweets}/{max_consecutive_old}")
            
            # 如果达到连续超时限制，停止滚动
            if reached_time_limit:
                self.logger.info(f"停止收集 - 连续超出 {hours_back} 小时时间限制")
                break
            
            # Check if we got new tweets
            if new_tweets_count == 0:
                no_new_tweets_count += 1
                self.logger.info(f"No new tweets found ({no_new_tweets_count} times)")
                
                # 如果连续3次没有新推文，停止
                if no_new_tweets_count >= 3:
                    self.logger.info("No new tweets for 3 scrolls, stopping...")
                    break
            else:
                no_new_tweets_count = 0  # 重置计数器
            
            # 检查导航状态，防止意外跳转到推文详情页
            current_url = self.page.url
            if '/status/' in current_url and 'home' not in current_url:
                self.logger.warning("检测到意外导航到推文详情页")
                recovery_success = await self._recover_from_accidental_navigation('https://x.com/home')
                if not recovery_success:
                    self.logger.error("无法从意外导航中恢复，停止收集")
                    break
            
            # Human-like scrolling
            await self._human_like_scroll()
            scroll_count += 1
            
            # 随机延迟
            delay = random.uniform(*self.delay_range)
            await asyncio.sleep(delay)
        
        self.logger.info(f"基于时间的收集完成：收集了 {len(tweets)} 条推文")
        return tweets
    
    async def _extract_tweets_from_page(self, timeline_source: str = 'unknown') -> List[Dict[str, Any]]:
        """Extract tweet data from current page with improved stability.
        
        Args:
            timeline_source: Source timeline ('following', 'for_you', or 'unknown')
        """
        tweets = []
        
        try:
            # 等待页面稳定
            self.logger.debug("Waiting for page to stabilize before extraction...")
            await asyncio.sleep(1.0)  # 额外等待确保页面稳定
            
            self.logger.debug("Looking for tweet elements on page...")
            
            # 尝试多种选择器来找推文
            tweet_selectors = [
                'article[data-testid="tweet"]',
                'div[data-testid="tweet"]', 
                'article[role="article"]',
                'div[data-testid="cellInnerDiv"] article'
            ]
            
            tweet_elements = []
            for selector in tweet_selectors:
                try:
                    # 增加超时时间到10秒
                    elements = await asyncio.wait_for(
                        self.page.locator(selector).all(), 
                        timeout=10.0
                    )
                    if elements:
                        tweet_elements = elements
                        self.logger.debug(f"Found tweet elements with selector: {selector}")
                        break
                except asyncio.TimeoutError:
                    self.logger.debug(f"Timeout finding elements with selector: {selector}")
                    continue
                except Exception as e:
                    self.logger.debug(f"Error with selector {selector}: {e}")
                    continue
            
            element_count = len(tweet_elements)
            self.logger.debug(f"Found {element_count} tweet elements total")
            
            # 分批处理推文元素，减少同时处理的压力
            batch_size = 5  # 每批处理5个推文
            for batch_start in range(0, element_count, batch_size):
                batch_end = min(batch_start + batch_size, element_count)
                batch_elements = tweet_elements[batch_start:batch_end]
                
                self.logger.debug(f"Processing tweet batch {batch_start//batch_size + 1}: elements {batch_start+1}-{batch_end}")
                
                # 并发处理批次内的推文
                batch_tasks = []
                for i, element in enumerate(batch_elements):
                    global_index = batch_start + i
                    task = self._extract_single_tweet_safe(element, global_index + 1, element_count, timeline_source)
                    batch_tasks.append(task)
                
                # 等待批次完成，设置合理超时
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=30.0  # 整个批次30秒超时
                    )
                    
                    # 收集有效结果
                    for result in batch_results:
                        if isinstance(result, dict) and result:
                            tweets.append(result)
                        elif isinstance(result, Exception):
                            self.logger.debug(f"Batch processing exception: {result}")
                            
                except asyncio.TimeoutError:
                    self.logger.warning(f"Batch {batch_start//batch_size + 1} processing timeout, continuing")
                    continue
                
                # 批次间短暂休息
                if batch_end < element_count:
                    await asyncio.sleep(0.2)
            
            self.logger.debug(f"Successfully extracted {len(tweets)} tweets from page")
            
        except Exception as e:
            self.logger.error(f"Failed to extract tweets from page: {e}")
        
        return tweets
    
    async def _extract_single_tweet_safe(self, element, index: int, total: int, timeline_source: str = 'unknown') -> Optional[Dict[str, Any]]:
        """安全地提取单个推文数据，包含容错处理
        
        Args:
            element: Tweet DOM element
            index: Current tweet index
            total: Total tweet count
            timeline_source: Source timeline ('following', 'for_you', or 'unknown')
        """
        try:
            self.logger.debug(f"Extracting data from tweet {index}/{total}...")
            
            # 简化的DOM连接检查，减少超时
            try:
                # 使用更简单的检查方式
                await element.is_visible()
            except Exception as e:
                self.logger.debug(f"Tweet {index} visibility check failed: {e}, trying anyway")
                # 不跳过，继续尝试提取
            
            # 提取推文数据，增加超时时间
            try:
                tweet_data = await asyncio.wait_for(
                    self._extract_tweet_data(element, timeline_source), 
                    timeout=15.0  # 增加到15秒超时
                )
                
                if tweet_data:
                    self.logger.debug(f"Successfully extracted tweet {index}: {tweet_data.get('id', 'unknown')}")
                    return tweet_data
                else:
                    self.logger.debug(f"Tweet {index} data extraction returned None")
                    return None
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout extracting tweet {index}, skipping")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to extract tweet {index} data: {e}")
            return None
    
    async def _extract_tweet_data(self, tweet_element, timeline_source: str = 'unknown') -> Optional[Dict[str, Any]]:
        """Extract data from a single tweet element with timeout protection.
        
        Args:
            tweet_element: The tweet DOM element to extract data from
            timeline_source: Source timeline ('following', 'for_you', or 'unknown')
        """
        try:
            # 提取推文的完整数据，包括长推文处理，每个步骤都有超时保护
            
            # 提取推文ID（快速操作，短超时）
            try:
                tweet_id = await asyncio.wait_for(
                    self._extract_tweet_id(tweet_element), 
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                self.logger.debug("Tweet ID extraction timeout, using fallback")
                import time
                tweet_id = f"tweet_{int(time.time() * 1000)}"
            
            # 提取推文文本（可能需要处理长推文，较长超时）
            try:
                tweet_text = await asyncio.wait_for(
                    self._extract_tweet_text(tweet_element), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.debug("Tweet text extraction timeout, using empty text")
                tweet_text = ""
            
            # 检查是否有引用推文（快速操作）
            try:
                quoted_tweet = await asyncio.wait_for(
                    self._extract_quoted_tweet(tweet_element), 
                    timeout=2.0
                )
                if quoted_tweet:
                    tweet_text += f"\n\n[引用推文: {quoted_tweet}]"
            except asyncio.TimeoutError:
                self.logger.debug("Quoted tweet extraction timeout, skipping")
                quoted_tweet = None
            
            # 提取作者信息（快速操作）
            try:
                author = await asyncio.wait_for(
                    self._extract_author_info(tweet_element), 
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                self.logger.debug("Author info extraction timeout, using default")
                author = {'name': 'Unknown', 'username': 'unknown', 'id': 'unknown', 'avatar_url': ''}
            
            # 提取时间戳和转发状态（快速操作）
            try:
                created_at, is_retweet = await asyncio.wait_for(
                    self._extract_timestamp(tweet_element), 
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                self.logger.debug("Timestamp extraction timeout, using current time")
                from datetime import datetime
                created_at = datetime.now().isoformat()
                is_retweet = False
            
            # 提取媒体URL（快速操作）
            try:
                media_urls = await asyncio.wait_for(
                    self._extract_media_urls(tweet_element), 
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                self.logger.debug("Media URLs extraction timeout, using empty list")
                media_urls = []
            
            # 提取互动数据（可能慢一些）
            try:
                engagement = await asyncio.wait_for(
                    self._extract_engagement_metrics(tweet_element), 
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                self.logger.debug("Engagement metrics extraction timeout, using zeros")
                engagement = {'likes': 0, 'retweets': 0, 'replies': 0, 'quotes': 0}
            
            tweet_data = {
                'id': tweet_id,
                'text': tweet_text,
                'author': author,
                'created_at': created_at,
                'media_urls': media_urls,
                'engagement': engagement,
                'quoted_tweet': quoted_tweet,
                'is_retweet': is_retweet,  # 添加转发标识
                'timeline_source': timeline_source  # 添加时间线来源标识
            }
            
            return tweet_data
            
        except Exception as e:
            self.logger.warning(f"Failed to extract tweet data: {e}")
            return None
    
    async def _extract_tweet_id(self, element) -> str:
        """Extract tweet ID from element."""
        try:
            # 尝试多种方式获取推文ID
            id_selectors = [
                # 查找推文链接
                'a[href*="/status/"]',
                '[data-testid="User-Name"] + div a[href*="/status/"]',
                'time[datetime] > a[href*="/status/"]',
                'time a[href*="/status/"]',
                '[data-testid="tweet"] a[href*="/status/"]',
                # 时间元素通常包含推文链接
                'time[datetime]',
                # 推文元素本身可能有ID属性
                '[data-testid="tweet"]'
            ]
            
            for selector in id_selectors:
                elements = element.locator(selector)
                count = await elements.count()
                
                for i in range(count):
                    elem = elements.nth(i)
                    
                    # 检查href属性
                    href = await elem.get_attribute('href')
                    if href and '/status/' in href:
                        parts = href.split('/status/')
                        if len(parts) > 1:
                            tweet_id = parts[1].split('?')[0].split('/')[0]  # 移除查询参数和路径
                            if tweet_id and tweet_id.isdigit():
                                self.logger.debug(f"Extracted tweet ID: {tweet_id}")
                                return tweet_id
                    
                    # 检查datetime属性（时间元素）
                    datetime_attr = await elem.get_attribute('datetime')
                    if datetime_attr:
                        # 时间元素的父元素通常是链接
                        parent = elem.locator('..')
                        parent_href = await parent.get_attribute('href')
                        if parent_href and '/status/' in parent_href:
                            parts = parent_href.split('/status/')
                            if len(parts) > 1:
                                tweet_id = parts[1].split('?')[0].split('/')[0]
                                if tweet_id and tweet_id.isdigit():
                                    self.logger.debug(f"Extracted tweet ID from time element: {tweet_id}")
                                    return tweet_id
            
            # 如果还找不到，尝试从页面URL获取
            current_url = self.page.url if self.page else ""
            if '/status/' in current_url:
                parts = current_url.split('/status/')
                if len(parts) > 1:
                    tweet_id = parts[1].split('?')[0].split('/')[0]
                    if tweet_id and tweet_id.isdigit():
                        return tweet_id
            
            # 生成带时间戳的ID以便调试
            import time
            timestamp_id = f"tweet_{int(time.time() * 1000)}"
            self.logger.warning(f"Could not extract real tweet ID, using: {timestamp_id}")
            return timestamp_id
            
        except Exception as e:
            self.logger.warning(f"Failed to extract tweet ID: {e}")
            import time
            return f"tweet_{int(time.time() * 1000)}"
    
    async def _extract_tweet_text(self, element) -> str:
        """Extract tweet text content, handling long tweets."""
        try:
            # 首先检查是否有"查看更多"或"Show more"按钮
            await self._expand_long_tweet(element)
            
            # 尝试多种选择器获取推文文本
            text_selectors = [
                '[data-testid="tweetText"]',
                'div[data-testid="tweetText"]',
                'span[data-testid="tweetText"]',
                # 推文文本通常在有lang属性的div中
                'div[lang]',
                'div[lang] span',
                '[data-testid="tweet"] div[lang]',
                'div[data-testid="tweetText"] span',
                # 更广泛的文本选择器
                '[data-testid="tweet"] span[lang]',
                'article div[lang]',
                # 备用选择器
                '[data-testid="tweet"] div:not([data-testid]) span',
                'article span:not([data-testid])'
            ]
            
            for selector in text_selectors:
                text_element = element.locator(selector)
                if await text_element.count() > 0:
                    # 获取所有匹配元素的文本并组合
                    all_elements = await text_element.all()
                    text_parts = []
                    for elem in all_elements:
                        text = await elem.inner_text()
                        if text.strip():
                            text_parts.append(text.strip())
                    
                    if text_parts:
                        full_text = ' '.join(text_parts)
                        self.logger.debug(f"Extracted tweet text (length: {len(full_text)}): {full_text[:100]}...")
                        return full_text
            
            # 如果以上都没找到，尝试获取整个推文区域的文本
            tweet_content = element.locator('div[data-testid="tweetText"]')
            if await tweet_content.count() > 0:
                return await tweet_content.inner_text()
                
        except Exception as e:
            self.logger.warning(f"Failed to extract tweet text: {e}")
        
        return ""
    
    async def _expand_long_tweet(self, element) -> None:
        """Expand long tweets by clicking '显示更多'/'Show more' buttons with case-insensitive matching and navigation protection."""
        try:
            # 记录当前URL，用于检测意外导航
            original_url = self.page.url
            
            # 调试：首先检查推文元素内的所有文本内容
            try:
                tweet_text = await element.inner_text()
                self.logger.debug(f"🔍 推文内容预览: {tweet_text[:200]}...")
                
                # 检查是否包含展开相关的文本
                if any(text in tweet_text.lower() for text in ['show more', '显示更多', '查看更多']):
                    self.logger.info(f"🎯 检测到包含展开文本的推文，开始查找按钮...")
                else:
                    self.logger.debug(f"📝 推文不包含展开文本，跳过按钮搜索")
                    return
            except Exception as e:
                self.logger.debug(f"获取推文文本失败: {e}")
            
            # 长推文展开按钮选择器 - 2025年7月更新版本，Twitter已移除role="button"
            show_more_selectors = [
                # === 新版本选择器：基于调试发现的实际DOM结构 ===
                # 不依赖role="button"，直接匹配span元素
                'span:text-is("显示更多"):not(a):not(a *)',
                'span:text-is("Show more"):not(a):not(a *)',
                'span:text-is("show more"):not(a):not(a *)',
                'span:text-is("SHOW MORE"):not(a):not(a *)',
                'span:text-is("Show More"):not(a):not(a *)',
                
                # === 基于CSS类的更精确匹配 ===
                'span.css-1jxf684:text-is("显示更多")',
                'span.css-1jxf684:text-is("Show more")',
                'span.css-1jxf684:text-is("show more")',
                'span.css-1jxf684:text-is("SHOW MORE")',
                'span.css-1jxf684:text-is("Show More")',
                
                # === 通过父元素定位 ===
                'div[lang] span:text-is("显示更多"):not(a):not(a *)',
                'div[lang] span:text-is("Show more"):not(a):not(a *)',
                'div[lang] span:text-is("show more"):not(a):not(a *)',
                
                # === 推文内容区域内 ===
                'div[data-testid="tweetText"] span:text-is("显示更多")',
                'div[data-testid="tweetText"] span:text-is("Show more")',
                'div[data-testid="tweetText"] span:text-is("show more")',
                
                # === 使用has-text的模糊匹配 ===
                'span:has-text("显示更多"):not(a):not(a *)',
                'span:has-text("show more"):not(a):not(a *)',
                'div[data-testid="tweetText"] span:has-text("显示更多")',
                'div[data-testid="tweetText"] span:has-text("show more")',
                'div[lang] span:has-text("显示更多"):not(a):not(a *)',
                'div[lang] span:has-text("show more"):not(a):not(a *)',
                
                # === 兼容旧版本的选择器（保留以防回滚） ===
                'span[role="button"]:text-is("显示更多"):not(a):not(a *)',
                'span[role="button"]:text-is("Show more"):not(a):not(a *)',
                'button:text-is("显示更多"):not(a):not(a *)',
                'button:text-is("Show more"):not(a):not(a *)'
            ]
            
            expanded_count = 0
            self.logger.info(f"🔍 开始尝试 {len(show_more_selectors)} 个选择器搜索展开按钮...")
            
            for i, selector in enumerate(show_more_selectors):
                show_more_button = element.locator(selector)
                button_count = await show_more_button.count()
                if button_count > 0:
                    self.logger.info(f"🔍 选择器 {i+1} 找到 {button_count} 个匹配: {selector[:80]}...")
                else:
                    self.logger.debug(f"🔍 选择器 {i+1} 无匹配: {selector[:80]}...")
                
                if button_count > 0:
                    button_text = await show_more_button.first.inner_text() if button_count > 0 else "unknown"
                    self.logger.info(f"🔍 发现长推文展开按钮 '{button_text}' (选择器: {selector[:50]}...)")
                    try:
                        # 使用最安全的点击方式：先检查元素属性
                        button_element = show_more_button.first
                        
                        # 验证元素不是链接
                        tag_name = await button_element.evaluate("el => el.tagName.toLowerCase()")
                        has_href = await button_element.evaluate("el => el.hasAttribute('href')")
                        is_visible = await button_element.is_visible()
                        is_enabled = await button_element.is_enabled()
                        
                        if tag_name == 'a' or has_href:
                            self.logger.debug(f"Skipping selector {selector} - element is a link")
                            continue
                            
                        if not is_visible or not is_enabled:
                            self.logger.debug(f"Skipping selector {selector} - element not visible/enabled")
                            continue
                        
                        # 使用安全的点击方式
                        self.logger.debug(f"Performing safe click on long tweet '{button_text}' button...")
                        await button_element.click(
                            timeout=3000, 
                            force=False,  # 不强制点击，确保元素可见
                            no_wait_after=False  # 等待导航完成
                        )
                        
                        # 立即检查导航状态
                        await asyncio.sleep(0.3)
                        current_url = self.page.url
                        if current_url != original_url:
                            self.logger.warning(f"Accidental navigation detected! From {original_url} to {current_url}")
                            # 立即返回原页面
                            await self._recover_from_accidental_navigation(original_url)
                            return
                        
                        # 等待内容展开
                        await asyncio.sleep(1.0)
                        expanded_count += 1
                        self.logger.info(f"✅ 成功展开长推文: '{button_text}' 按钮点击成功")
                        
                        # 继续检查是否还有其他展开按钮
                        continue
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to click long tweet expansion button '{button_text}' with selector {selector}: {e}")
                        # 检查是否因为意外导航而失败
                        current_url = self.page.url
                        if current_url != original_url:
                            self.logger.warning(f"Navigation occurred during click attempt: {current_url}")
                            await self._recover_from_accidental_navigation(original_url)
                            return
                        continue
            
            if expanded_count > 0:
                self.logger.info(f"🎯 本条推文总共展开了 {expanded_count} 个长推文按钮")
            else:
                # 如果没找到按钮，但推文包含展开文本，那么调试DOM结构
                try:
                    tweet_text = await element.inner_text()
                    if any(text in tweet_text.lower() for text in ['show more', '显示更多', '查看更多']):
                        self.logger.warning("⚠️ 推文包含展开文本但未找到按钮，开始调试DOM结构...")
                        
                        # 查找所有可能的按钮元素
                        all_buttons = element.locator('button, [role="button"], span[role="button"]')
                        button_count = await all_buttons.count()
                        self.logger.info(f"🔍 推文中总共找到 {button_count} 个按钮/可点击元素")
                        
                        for j in range(min(button_count, 5)):  # 只检查前5个
                            try:
                                btn = all_buttons.nth(j)
                                btn_text = await btn.inner_text()
                                btn_html = await btn.inner_html()
                                self.logger.info(f"   按钮 {j+1}: 文本='{btn_text[:50]}' HTML='{btn_html[:100]}...'")
                            except:
                                pass
                    else:
                        self.logger.debug("📄 本条推文未发现需要展开的长推文按钮")
                except Exception as e:
                    self.logger.debug(f"调试DOM结构失败: {e}")
                        
        except Exception as e:
            self.logger.debug(f"Error in _expand_long_tweet: {e}")
            # 检查是否因为导航问题导致的错误
            try:
                current_url = self.page.url
                if '/status/' in current_url and '/home' not in current_url:
                    self.logger.warning(f"Detected navigation to tweet detail page: {current_url}")
                    await self._recover_from_accidental_navigation('https://x.com/home')
            except:
                pass
    
    async def _extract_quoted_tweet(self, element) -> Optional[str]:
        """Extract quoted tweet content if present."""
        try:
            # 查找引用推文的常见选择器
            quoted_tweet_selectors = [
                'div[data-testid="quote"]',
                'div[data-testid="quoteTweet"]',
                'div[role="blockquote"]',
                'blockquote',
                # 引用推文通常在特定的容器中
                'div[data-testid="tweet"] div[role="blockquote"]',
                'article div[data-testid="quote"]'
            ]
            
            for selector in quoted_tweet_selectors:
                quoted_element = element.locator(selector)
                if await quoted_element.count() > 0:
                    # 提取引用推文的文本
                    quoted_text = await quoted_element.inner_text()
                    if quoted_text.strip():
                        self.logger.debug(f"Found quoted tweet: {quoted_text[:100]}...")
                        return quoted_text.strip()
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting quoted tweet: {e}")
            return None
    
    async def _extract_author_info(self, element) -> Dict[str, str]:
        """Extract author information."""
        try:
            # 尝试多种方式提取作者信息
            author_selectors = [
                '[data-testid="User-Name"]',
                '[data-testid="UserName"]', 
                'div[data-testid="User-Name"]',
                'span[data-testid="User-Name"]'
            ]
            
            username = "unknown"
            display_name = "Unknown"
            
            for selector in author_selectors:
                user_element = element.locator(selector)
                if await user_element.count() > 0:
                    # 提取用户名（从链接）
                    username_links = user_element.locator('a[href^="/"]')
                    if await username_links.count() > 0:
                        for i in range(await username_links.count()):
                            link = username_links.nth(i)
                            href = await link.get_attribute('href')
                            if href and href.startswith('/') and not '/status/' in href:
                                username = href.strip('/').split('/')[0]
                                if username and username != 'home' and username != 'search':
                                    self.logger.debug(f"Extracted username: {username}")
                                    break
                    
                    # 提取显示名称（第一个粗体文本通常是显示名称）
                    name_elements = user_element.locator('span')
                    if await name_elements.count() > 0:
                        for i in range(await name_elements.count()):
                            span = name_elements.nth(i)
                            text = await span.inner_text()
                            if text.strip() and not text.startswith('@') and len(text.strip()) > 1:
                                display_name = text.strip()
                                self.logger.debug(f"Extracted display name: {display_name}")
                                break
                    
                    # 如果找到了有效信息就退出
                    if username != "unknown" or display_name != "Unknown":
                        break
            
            # 如果还是没找到，尝试其他选择器
            if username == "unknown":
                # 查找任何指向用户档案的链接
                profile_links = element.locator('a[href^="/"]')
                if await profile_links.count() > 0:
                    for i in range(min(5, await profile_links.count())):  # 只检查前5个链接
                        link = profile_links.nth(i)
                        href = await link.get_attribute('href')
                        if href and href.startswith('/') and not any(x in href for x in ['/status/', '/search', '/home', '/i/']):
                            potential_username = href.strip('/').split('/')[0]
                            if potential_username and len(potential_username) > 1:
                                username = potential_username
                                self.logger.debug(f"Extracted username from profile link: {username}")
                                break
            
            # 查找头像来验证作者信息
            avatar_url = ""
            avatar_selectors = [
                'img[src*="profile_images"]',
                '[data-testid="UserAvatar-Container-"] img',
                'img[alt*="profile"]'
            ]
            
            for selector in avatar_selectors:
                avatar_element = element.locator(selector)
                if await avatar_element.count() > 0:
                    src = await avatar_element.first.get_attribute('src')
                    if src:
                        avatar_url = src
                        break
            
            return {
                'name': display_name,
                'username': username,
                'id': username,
                'avatar_url': avatar_url
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to extract author info: {e}")
            return {
                'name': 'Unknown',
                'username': 'unknown', 
                'id': 'unknown',
                'avatar_url': ''
            }
    
    async def _extract_timestamp(self, element) -> tuple[str, bool]:
        """Extract tweet timestamp and determine if it's a retweet.
        
        Returns:
            tuple[str, bool]: (timestamp, is_retweet)
            - For original tweets: returns (original_time, False)
            - For retweets: returns (retweet_time, True) - prioritizes retweet time over original
        """
        try:
            # First check if this is a retweet by looking for retweet indicators
            is_retweet = await self._is_retweet(element)
            
            if is_retweet:
                # For retweets, try to get the retweet time (newer time) first
                retweet_time = await self._extract_retweet_timestamp(element)
                if retweet_time:
                    self.logger.debug(f"Extracted retweet timestamp: {retweet_time}")
                    return retweet_time, True
                else:
                    self.logger.debug("Failed to extract retweet time, falling back to original time")
            
            # Extract original tweet time (fallback for retweets or primary for original tweets)
            original_time = await self._extract_original_timestamp(element)
            if original_time:
                return original_time, is_retweet
                    
            # Final fallback to current time
            return datetime.now().isoformat(), is_retweet
            
        except Exception as e:
            self.logger.warning(f"Failed to extract timestamp: {e}")
            return datetime.now().isoformat(), False
    
    async def _is_retweet(self, element) -> bool:
        """Check if the tweet element is a retweet."""
        try:
            # Look for retweet indicators in the tweet
            retweet_indicators = [
                # Text indicators
                ':has-text("Retweeted")',
                ':has-text("转推了")',
                ':has-text("reposted")',
                ':has-text("转发了")',
                # Icon indicators
                '[data-testid="retweet"]',
                'svg[aria-label*="Retweet"]',
                'svg[aria-label*="转推"]',
                # Structure indicators - retweets often have specific DOM patterns
                '[data-testid="socialContext"]',
                # User action indicators
                'span:has-text("retweeted")',
                'span:has-text("转推了")',
                # Look for "X retweeted" or "X 转推了" patterns
                'div:has-text("retweeted")',
                'div:has-text("转推了")',
            ]
            
            for indicator in retweet_indicators:
                indicator_element = element.locator(indicator)
                if await indicator_element.count() > 0:
                    # Additional verification - make sure it's not just the retweet button
                    text_content = await indicator_element.first.inner_text()
                    if text_content and any(keyword in text_content.lower() for keyword in ['retweeted', '转推了', 'reposted', '转发了']):
                        self.logger.debug(f"Retweet detected with indicator: {indicator}")
                        return True
            
            # Alternative method: check if there are multiple time elements (original + retweet)
            time_elements = element.locator('time[datetime]')
            time_count = await time_elements.count()
            if time_count > 1:
                self.logger.debug(f"Multiple time elements found ({time_count}), likely a retweet")
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking retweet status: {e}")
            return False
    
    async def _extract_retweet_timestamp(self, element) -> Optional[str]:
        """Extract the retweet timestamp (when the retweet action occurred)."""
        try:
            # Strategy 1: Look for time elements in retweet context areas
            retweet_time_selectors = [
                # Time in social context area (where "X retweeted" appears)
                '[data-testid="socialContext"] time[datetime]',
                '[data-testid="socialContext"] ~ * time[datetime]',
                # Time elements associated with retweet indicators
                ':has-text("retweeted") time[datetime]',
                ':has-text("转推了") time[datetime]',
                # Sometimes the newer time is the first one in DOM order
                'time[datetime]:first-child',
            ]
            
            for selector in retweet_time_selectors:
                time_element = element.locator(selector)
                if await time_element.count() > 0:
                    datetime_attr = await time_element.first.get_attribute('datetime')
                    if datetime_attr:
                        self.logger.debug(f"Found retweet time with selector: {selector}")
                        return datetime_attr
            
            # Strategy 2: If there are multiple time elements, assume the first/latest one is retweet time
            all_time_elements = element.locator('time[datetime]')
            time_count = await all_time_elements.count()
            
            if time_count > 1:
                # Get all timestamps and find the most recent one (likely the retweet time)
                timestamps = []
                for i in range(time_count):
                    time_elem = all_time_elements.nth(i)
                    datetime_attr = await time_elem.get_attribute('datetime')
                    if datetime_attr:
                        try:
                            parsed_time = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                            timestamps.append((datetime_attr, parsed_time))
                        except:
                            continue
                
                if timestamps:
                    # Sort by time and return the most recent (retweet time)
                    timestamps.sort(key=lambda x: x[1], reverse=True)
                    newest_timestamp = timestamps[0][0]
                    self.logger.debug(f"Selected newest timestamp as retweet time: {newest_timestamp}")
                    return newest_timestamp
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting retweet timestamp: {e}")
            return None
    
    async def _extract_original_timestamp(self, element) -> Optional[str]:
        """Extract the original tweet timestamp."""
        try:
            # Look for time element - use more specific selectors to avoid duplicates
            time_selectors = [
                'time[datetime]',  # 有datetime属性的time元素
                'a time[datetime]',  # 链接内的time元素
                'time:last-child',  # 对于转发，原推文时间通常在后面
                'time'  # 兜底选择器
            ]
            
            for selector in time_selectors:
                time_element = element.locator(selector)
                if await time_element.count() > 0:
                    try:
                        datetime_attr = await time_element.first.get_attribute('datetime')
                        if datetime_attr:
                            return datetime_attr
                    except Exception as e:
                        self.logger.debug(f"Time selector '{selector}' failed: {e}")
                        continue
                    
            # 如果所有方法都失败，使用JavaScript提取
            timestamp = await element.evaluate("""
                (el) => {
                    const timeElements = el.querySelectorAll('time[datetime]');
                    if (timeElements.length > 0) {
                        // For retweets, the original time is usually the last one
                        return timeElements[timeElements.length - 1].getAttribute('datetime');
                    }
                    return null;
                }
            """)
            
            return timestamp
                    
        except Exception as e:
            self.logger.debug(f"Error extracting original timestamp: {e}")
            return None
    
    async def _extract_media_urls(self, element) -> List[str]:
        """Extract media URLs from tweet."""
        try:
            media_urls = []
            
            # Extract image URLs
            img_elements = element.locator('img[src*="pbs.twimg.com"]')
            img_count = await img_elements.count()
            for i in range(img_count):
                img_element = img_elements.nth(i)
                src = await img_element.get_attribute('src')
                if src and src not in media_urls:
                    media_urls.append(src)
            
            # Extract video thumbnails and URLs
            video_elements = element.locator('video source')
            video_count = await video_elements.count()
            for i in range(video_count):
                video_element = video_elements.nth(i)
                src = await video_element.get_attribute('src')
                if src and src not in media_urls:
                    media_urls.append(src)
            
            return media_urls
        except Exception as e:
            self.logger.warning(f"Failed to extract media URLs: {e}")
            return []
    
    async def _extract_engagement_metrics(self, element) -> Dict[str, int]:
        """Extract engagement metrics (likes, retweets, etc.)."""
        try:
            metrics = {
                'likes': 0,
                'retweets': 0,
                'replies': 0,
                'quotes': 0
            }
            
            # 互动按钮的多种选择器
            engagement_selectors = {
                'replies': [
                    '[data-testid="reply"]',
                    '[aria-label*="reply"]', 
                    '[aria-label*="Reply"]',
                    'button[aria-label*="回复"]'
                ],
                'retweets': [
                    '[data-testid="retweet"]',
                    '[aria-label*="retweet"]',
                    '[aria-label*="Retweet"]', 
                    'button[aria-label*="转推"]'
                ],
                'likes': [
                    '[data-testid="like"]',
                    '[aria-label*="like"]',
                    '[aria-label*="Like"]',
                    'button[aria-label*="喜欢"]'
                ],
                'quotes': [
                    '[data-testid="quote"]',
                    '[aria-label*="quote"]',
                    '[aria-label*="Quote"]'
                ]
            }
            
            for metric_type, selectors in engagement_selectors.items():
                for selector in selectors:
                    buttons = element.locator(selector)
                    if await buttons.count() > 0:
                        try:
                            # 尝试从按钮文本获取数字
                            button_text = await buttons.first.inner_text()
                            if button_text.strip():
                                number = self._parse_number(button_text)
                                if number > 0:
                                    metrics[metric_type] = number
                                    self.logger.debug(f"Extracted {metric_type}: {number}")
                                    break
                            
                            # 尝试从aria-label获取数字
                            aria_label = await buttons.first.get_attribute('aria-label')
                            if aria_label:
                                # 从aria-label中提取数字
                                import re
                                numbers = re.findall(r'\d+', aria_label)
                                if numbers:
                                    number = int(numbers[0])
                                    metrics[metric_type] = number
                                    self.logger.debug(f"Extracted {metric_type} from aria-label: {number}")
                                    break
                                    
                        except Exception as e:
                            self.logger.debug(f"Failed to extract {metric_type}: {e}")
                            continue
            
            # 如果还没找到，尝试查找包含数字的span元素
            if all(v == 0 for v in metrics.values()):
                # 使用JavaScript来查找包含数字的span元素，因为CSS选择器不支持正则表达式
                number_texts = await element.evaluate("""
                    (el) => {
                        const spans = el.querySelectorAll('span');
                        const numberTexts = [];
                        spans.forEach(span => {
                            const text = span.textContent?.trim();
                            if (text && /^\\d+/.test(text)) {
                                numberTexts.push(text);
                            }
                        });
                        return numberTexts;
                    }
                """)
                span_count = len(number_texts)
                
                for i, text in enumerate(number_texts[:4]):  # 最多检查4个数字文本
                    number = self._parse_number(text)
                    
                    if number > 0:
                        # 根据位置推测是什么类型的互动
                        if i == 0:
                            metrics['replies'] = number
                        elif i == 1:
                            metrics['retweets'] = number
                        elif i == 2:
                            metrics['likes'] = number
                        elif i == 3:
                            metrics['quotes'] = number
            
            self.logger.debug(f"Final engagement metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to extract engagement metrics: {e}")
            return {
                'likes': 0,
                'retweets': 0,
                'replies': 0,
                'quotes': 0
            }
    
    def _parse_number(self, text: str) -> int:
        """Parse number from text (handles K, M suffixes)."""
        try:
            text = text.strip().replace(',', '')
            if 'K' in text:
                return int(float(text.replace('K', '')) * 1000)
            elif 'M' in text:
                return int(float(text.replace('M', '')) * 1000000)
            else:
                return int(text) if text.isdigit() else 0
        except:
            return 0
    
    def _parse_tweet_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse tweet timestamp and convert to local timezone for comparison.
        
        Twitter timestamps are usually in ISO format with UTC timezone (Z suffix).
        We need to convert them to local timezone for proper comparison.
        
        Args:
            timestamp_str: ISO format timestamp string (e.g., "2025-07-31T11:40:50.000Z")
            
        Returns:
            datetime object in local timezone (timezone-naive for comparison)
        """
        try:
            # Parse the ISO timestamp
            tweet_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Convert to local timezone
            local_tweet_time = tweet_time.astimezone()
            
            # Remove timezone info to make it naive for comparison with naive cutoff_time
            local_tweet_time_naive = local_tweet_time.replace(tzinfo=None)
            
            self.logger.debug(f"Parsed timestamp: {timestamp_str} -> UTC: {tweet_time} -> Local: {local_tweet_time_naive}")
            
            return local_tweet_time_naive
            
        except Exception as e:
            self.logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
            # Fallback to current time
            return datetime.now()
    
    def _adjust_scroll_params_by_mode(self):
        """根据滚动模式调整参数"""
        if self.scroll_mode == 'fast':
            # 快速模式：更大滚动距离，更短等待时间
            self.scroll_distance_range = [3000, 4500]
            self.content_wait_range = [1, 2]
            self.bottom_wait_range = [1, 3]
            self.logger.info("🚀 滚动模式：快速模式 - 提升收集速度，但可能增加检测风险")
        elif self.scroll_mode == 'safe':
            # 安全模式：保守滚动距离，更长等待时间
            self.scroll_distance_range = [1200, 2000]
            self.content_wait_range = [3, 5]
            self.bottom_wait_range = [4, 8]
            self.logger.info("🛡️ 滚动模式：安全模式 - 降低检测风险，但收集速度较慢")
        else:
            # 平衡模式：使用配置文件中的设置
            self.logger.info("⚖️ 滚动模式：平衡模式 - 兼顾速度和安全性")
    
    async def _human_like_scroll(self) -> None:
        """Perform configurable human-like scrolling optimized for Twitter's infinite scroll."""
        try:
            # 记录滚动前的页面高度
            pre_scroll_height = await self.page.evaluate("document.documentElement.scrollHeight")
            
            # 使用可配置的滚动距离
            scroll_distance = random.randint(*self.scroll_distance_range)
            scroll_info = f"滚动距离: {scroll_distance}px (模式: {self.scroll_mode})"
            
            # 滚动到页面底部附近触发更多内容加载
            await self.page.evaluate(f"""
                window.scrollBy({{
                    top: {scroll_distance},
                    behavior: 'smooth'
                }});
            """)
            
            # 使用可配置的内容加载等待
            content_wait = random.uniform(*self.content_wait_range)
            self.logger.debug(f"{scroll_info}, 等待内容加载: {content_wait:.1f}s")
            await asyncio.sleep(content_wait)
            
            # 检查是否接近页面底部，如果是则额外等待
            is_near_bottom = await self.page.evaluate("""
                () => {
                    const windowHeight = window.innerHeight;
                    const documentHeight = document.documentElement.scrollHeight;
                    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                    return (scrollTop + windowHeight) > (documentHeight * 0.8);
                }
            """)
            
            if is_near_bottom:
                bottom_wait = random.uniform(*self.bottom_wait_range)
                self.logger.debug(f"接近页面底部，额外等待: {bottom_wait:.1f}s")
                await asyncio.sleep(bottom_wait)
            
            # 等待新内容加载完成
            await self._wait_for_new_content(pre_scroll_height)
            
        except Exception as e:
            self.logger.warning(f"Scrolling failed: {e}")
    
    async def _wait_for_new_content(self, previous_height: int, max_wait: float = 8.0) -> bool:
        """等待新内容加载完成，返回是否有新内容"""
        try:
            start_time = asyncio.get_event_loop().time()
            check_interval = 0.5
            
            while (asyncio.get_event_loop().time() - start_time) < max_wait:
                current_height = await self.page.evaluate("document.documentElement.scrollHeight")
                
                if current_height > previous_height:
                    self.logger.debug(f"检测到新内容：页面高度 {previous_height} -> {current_height}")
                    # 新内容加载后再等待一小段时间确保渲染完成
                    await asyncio.sleep(0.8)
                    return True
                
                await asyncio.sleep(check_interval)
            
            self.logger.debug(f"等待 {max_wait}s 后无新内容加载")
            return False
            
        except Exception as e:
            self.logger.debug(f"等待新内容失败: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up browser resources."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            self.logger.info("Twitter scraper cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")
    
    async def _recover_from_accidental_navigation(self, target_url: str) -> bool:
        """Recover from accidental navigation to tweet detail pages while preserving scroll position."""
        try:
            current_url = self.page.url
            self.logger.info(f"Attempting to recover from accidental navigation: {current_url}")
            
            # 方法1: 使用浏览器后退按钮（优先选择，保持滚动位置）
            if '/status/' in current_url or '/photo/' in current_url:
                try:
                    self.logger.info("Using browser back button to preserve scroll position...")
                    await self.page.go_back(wait_until='domcontentloaded', timeout=10000)
                    await asyncio.sleep(3)  # 给页面更多时间加载
                    
                    # 检查是否成功返回到主页时间线
                    recovered_url = self.page.url
                    if '/home' in recovered_url:
                        self.logger.info("✅ Successfully recovered using browser back button (scroll position preserved)")
                        
                        # 检查是否需要重新激活Following标签
                        if 'following' in target_url.lower():
                            await asyncio.sleep(2)
                            # 验证Following标签是否仍然激活
                            is_following_active = await self._verify_following_timeline()
                            if not is_following_active:
                                self.logger.info("Re-activating Following timeline after recovery...")
                                await self._navigate_to_following_timeline()
                        
                        return True
                    else:
                        self.logger.warning(f"Back button led to unexpected page: {recovered_url}")
                        
                except Exception as e:
                    self.logger.debug(f"Browser back button failed: {e}")
            
            # 方法2: 键盘快捷键返回（Alt+Left Arrow 或 Cmd+Left Arrow）
            try:
                self.logger.info("Trying keyboard shortcut to go back...")
                import platform
                if platform.system() == "Darwin":  # macOS
                    await self.page.keyboard.press("Meta+ArrowLeft")
                else:  # Windows/Linux
                    await self.page.keyboard.press("Alt+ArrowLeft")
                
                await asyncio.sleep(3)
                
                current_url_after_shortcut = self.page.url
                if '/home' in current_url_after_shortcut and '/status/' not in current_url_after_shortcut:
                    self.logger.info("✅ Successfully recovered using keyboard shortcut (scroll position preserved)")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"Keyboard shortcut failed: {e}")
            
            # 方法3: JavaScript历史记录返回
            try:
                self.logger.info("Trying JavaScript history.back()...")
                await self.page.evaluate("window.history.back()")
                await asyncio.sleep(3)
                
                current_url_after_js = self.page.url
                if '/home' in current_url_after_js and '/status/' not in current_url_after_js:
                    self.logger.info("✅ Successfully recovered using JavaScript history.back() (scroll position preserved)")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"JavaScript history.back() failed: {e}")
            
            # 方法4: 最后手段 - 直接导航（会丢失滚动位置）
            self.logger.warning("⚠️  All scroll-preserving methods failed, using direct navigation (will lose scroll position)")
            try:
                await self.page.goto(target_url, wait_until='domcontentloaded', timeout=15000)
                await asyncio.sleep(3)
                
                # 验证是否成功导航
                final_url = self.page.url
                if '/home' in final_url:
                    self.logger.info(f"Successfully recovered by navigating to {target_url} (scroll position lost)")
                    
                    # 如果目标是Following页面，需要重新切换到Following标签
                    if 'following' in target_url.lower():
                        await asyncio.sleep(2)
                        await self._navigate_to_following_timeline()
                    
                    return True
                else:
                    self.logger.warning(f"Navigation recovery failed: ended up at {final_url}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Failed to recover navigation: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False
    
    async def _check_and_recover_navigation(self) -> bool:
        """Check if we're still on the correct page and recover if needed."""
        try:
            current_url = self.page.url
            
            # 如果在推文详情页，需要恢复
            if '/status/' in current_url and '/home' not in current_url:
                self.logger.warning(f"Detected navigation to tweet detail page: {current_url}")
                
                # 尝试恢复到home页面
                recovery_success = await self._recover_from_accidental_navigation('https://x.com/home')
                
                if not recovery_success:
                    self.logger.error("Failed to recover from accidental navigation")
                    return False
                
                return True
            
            # 如果在home页面，检查是否需要切换到Following标签
            elif '/home' in current_url:
                return True
            
            # 其他页面也尝试恢复
            else:
                self.logger.warning(f"Unexpected page detected: {current_url}")
                return await self._recover_from_accidental_navigation('https://x.com/home')
                
        except Exception as e:
            self.logger.error(f"Navigation check failed: {e}")
            return False
    
    async def _navigate_to_following_timeline(self) -> bool:
        """
        Navigate to Following timeline with improved logic.
        Since both For You and Following use the same URL (https://x.com/home),
        we need to click the Following tab to switch the timeline.
        
        Returns:
            True if navigation successful, False otherwise
        """
        self.logger.info("Navigating to Following timeline...")
        
        try:
            # First ensure we're on the home page
            current_url = self.page.url
            if '/home' not in current_url:
                self.logger.info("Not on home page, navigating there first...")
                await self.page.goto('https://x.com/home', wait_until='domcontentloaded', timeout=30000)
                await asyncio.sleep(5)
            
            # Wait for the timeline tabs to load
            self.logger.info("Waiting for timeline navigation to load...")
            try:
                await self.page.wait_for_selector('[role="tablist"]', timeout=15000)
                await asyncio.sleep(3)  # Additional wait for JavaScript to render tabs
                self.logger.info("Timeline navigation loaded")
            except:
                self.logger.warning("Timeline navigation tabs not found, continuing...")
            
            # Strategy 1: Try direct tab selection with multiple approaches
            following_clicked = await self._try_click_following_tab()
            
            if following_clicked:
                # Verify we're on Following timeline by checking page content
                await asyncio.sleep(5)  # Wait for content to load
                
                # Check if the Following tab is active/selected
                is_following_active = await self._verify_following_timeline()
                if is_following_active:
                    self.logger.info("Successfully navigated to Following timeline")
                    return True
                else:
                    self.logger.warning("Following tab clicked but timeline may not have switched")
            
            # Strategy 2: If direct clicking failed, try JavaScript approach
            self.logger.info("Trying JavaScript-based Following navigation...")
            js_success = await self._javascript_click_following()
            
            if js_success:
                await asyncio.sleep(5)
                if await self._verify_following_timeline():
                    self.logger.info("JavaScript Following navigation successful")
                    return True
            
            # Strategy 3: Keyboard navigation as last resort
            self.logger.info("Trying keyboard navigation to Following...")
            keyboard_success = await self._keyboard_navigate_following()
            
            if keyboard_success:
                await asyncio.sleep(5)
                if await self._verify_following_timeline():
                    self.logger.info("Keyboard Following navigation successful")
                    return True
            
            self.logger.warning("All Following navigation strategies failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Following timeline navigation failed: {e}")
            return False
    
    async def _try_click_following_tab(self) -> bool:
        """Try to click the Following tab using various selectors."""
        following_selectors = [
            # Most specific selectors first (support multiple languages)
            '[role="tablist"] [role="tab"]:has-text("Following")',
            '[role="tablist"] [role="tab"]:has-text("正在关注")',
            '[role="tablist"] [role="tab"]:has-text("关注")',
            '[role="tablist"] div:has-text("Following")',
            '[role="tablist"] div:has-text("正在关注")',
            '[role="tablist"] div:has-text("关注")',
            # Broader selectors
            '[role="tab"]:has-text("Following")',
            '[role="tab"]:has-text("正在关注")',
            '[role="tab"]:has-text("关注")',
            'a:has-text("Following")',
            'a:has-text("关注")',
            'button:has-text("Following")',
            'button:has-text("关注")',
            # Navigation specific
            'nav [role="tab"]:has-text("Following")',
            'nav [role="tab"]:has-text("关注")',
            'div[data-testid="primaryColumn"] [role="tab"]:has-text("Following")',
            'div[data-testid="primaryColumn"] [role="tab"]:has-text("关注")',
            # Aria labels
            '[aria-label*="Following"]',
            '[aria-label*="正在关注"]',
            '[aria-label*="关注"]'
        ]
        
        for selector in following_selectors:
            try:
                element = self.page.locator(selector)
                count = await element.count()
                
                if count > 0:
                    self.logger.info(f"Found Following tab with selector: {selector}")
                    await element.first.click()
                    self.logger.info("Following tab clicked successfully")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"Selector '{selector}' failed: {e}")
                continue
        
        return False
    
    async def _javascript_click_following(self) -> bool:
        """Use JavaScript to find and click the Following tab."""
        try:
            result = await self.page.evaluate("""
                () => {
                    // First, let's debug what tabs are actually available
                    const allTabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    const tabInfo = allTabs.map(tab => ({
                        text: tab.textContent?.trim(),
                        selected: tab.getAttribute('aria-selected'),
                        element: tab.tagName
                    }));
                    
                    // Look for Following tab more precisely (support multiple languages)
                    const followingTab = allTabs.find(tab => {
                        const text = tab.textContent?.trim();
                        return text === 'Following' || text === '正在关注' || text === '关注';
                    });
                    
                    if (followingTab) {
                        // Try multiple click methods
                        try {
                            // Method 1: Direct click
                            followingTab.click();
                            return { 
                                success: true, 
                                method: 'direct_click',
                                tabInfo: tabInfo,
                                clicked_text: followingTab.textContent?.trim()
                            };
                        } catch (e1) {
                            try {
                                // Method 2: Mouse event
                                const clickEvent = new MouseEvent('click', { 
                                    bubbles: true, 
                                    cancelable: true,
                                    view: window
                                });
                                followingTab.dispatchEvent(clickEvent);
                                return { 
                                    success: true, 
                                    method: 'mouse_event',
                                    tabInfo: tabInfo,
                                    clicked_text: followingTab.textContent?.trim()
                                };
                            } catch (e2) {
                                try {
                                    // Method 3: Focus and enter
                                    followingTab.focus();
                                    const enterEvent = new KeyboardEvent('keydown', { key: 'Enter' });
                                    followingTab.dispatchEvent(enterEvent);
                                    return { 
                                        success: true, 
                                        method: 'keyboard_enter',
                                        tabInfo: tabInfo,
                                        clicked_text: followingTab.textContent?.trim()
                                    };
                                } catch (e3) {
                                    return { 
                                        success: false, 
                                        error: `All click methods failed: ${e1.message}, ${e2.message}, ${e3.message}`,
                                        tabInfo: tabInfo
                                    };
                                }
                            }
                        }
                    }
                    
                    return { 
                        success: false, 
                        error: 'Following tab not found',
                        tabInfo: tabInfo
                    };
                }
            """)
            
            # Log debug information
            if 'tabInfo' in result:
                self.logger.info(f"Available tabs: {result['tabInfo']}")
            
            if result.get('success'):
                self.logger.info(f"JavaScript Following click successful via {result.get('method')}, clicked: {result.get('clicked_text')}")
                return True
            else:
                self.logger.warning(f"JavaScript Following click failed: {result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"JavaScript Following navigation error: {e}")
            return False
    
    async def _keyboard_navigate_following(self) -> bool:
        """Use keyboard navigation to reach the Following tab."""
        try:
            # Focus on the page first
            await self.page.keyboard.press('Tab')
            await asyncio.sleep(1)
            
            # Try to navigate to tabs area and then move right
            for _ in range(3):  # Try a few tab navigation attempts
                await self.page.keyboard.press('Tab')
                await asyncio.sleep(0.5)
            
            # Try arrow key navigation (assuming we're in the tab area)
            await self.page.keyboard.press('ArrowRight')
            await asyncio.sleep(1)
            await self.page.keyboard.press('Enter')
            await asyncio.sleep(2)
            
            return True  # Assume success, will be verified by caller
            
        except Exception as e:
            self.logger.error(f"Keyboard Following navigation error: {e}")
            return False
    
    async def _verify_following_timeline(self) -> bool:
        """Verify that we're currently viewing the Following timeline."""
        try:
            # Give page time to update after clicking
            await asyncio.sleep(2)
            
            # Strategy 1: Check if Following tab appears selected/active
            active_selectors = [
                '[role="tablist"] [role="tab"][aria-selected="true"]:has-text("Following")',
                '[role="tablist"] [role="tab"][aria-selected="true"]:has-text("正在关注")',
                '[role="tablist"] [role="tab"][aria-selected="true"]:has-text("关注")',
                '[role="tablist"] [role="tab"].selected:has-text("Following")',
                '[role="tablist"] [role="tab"].selected:has-text("关注")',
                '[role="tab"][data-testid*="following"]',
            ]
            
            for selector in active_selectors:
                element = self.page.locator(selector)
                if await element.count() > 0:
                    self.logger.info(f"Following tab is active (selector: {selector})")
                    return True
            
            # Strategy 2: Use JavaScript to check DOM state with detailed debugging
            result = await self.page.evaluate("""
                () => {
                    const tabElements = Array.from(document.querySelectorAll('[role="tab"]'));
                    const tabStates = tabElements.map(tab => ({
                        text: tab.textContent?.trim(),
                        selected: tab.getAttribute('aria-selected'),
                        classes: tab.className,
                        href: tab.href || null
                    }));
                    
                    // Look for active Following tab (support multiple languages)
                    const followingTab = tabElements.find(tab => 
                        tab.textContent?.includes('Following') || 
                        tab.textContent?.includes('正在关注') || 
                        tab.textContent?.includes('关注')
                    );
                    
                    const forYouTab = tabElements.find(tab => 
                        tab.textContent?.includes('For you') || 
                        tab.textContent?.includes('为你推荐')
                    );
                    
                    return {
                        tabStates: tabStates,
                        followingFound: !!followingTab,
                        followingActive: followingTab ? (
                            followingTab.getAttribute('aria-selected') === 'true' ||
                            followingTab.classList.contains('selected') ||
                            followingTab.classList.contains('active')
                        ) : false,
                        forYouFound: !!forYouTab,
                        forYouActive: forYouTab ? (
                            forYouTab.getAttribute('aria-selected') === 'true' ||
                            forYouTab.classList.contains('selected') ||
                            forYouTab.classList.contains('active')
                        ) : false
                    };
                }
            """)
            
            self.logger.info(f"Tab verification result: {result}")
            
            # Only consider it successful if Following is active AND For You is NOT active
            if result.get('followingActive') and not result.get('forYouActive'):
                self.logger.info("Following timeline verified - Following active, For You inactive")
                return True
            elif result.get('forYouActive'):
                self.logger.warning("Still on For You timeline - Following navigation failed")
                return False
            else:
                self.logger.warning("Cannot determine active timeline state")
                return False
            
        except Exception as e:
            self.logger.warning(f"Following timeline verification error: {e}")
            return False
    
    async def _navigate_to_for_you_timeline(self) -> bool:
        """Navigate to For You timeline (usually the default)."""
        self.logger.info("Navigating to For You timeline...")
        
        try:
            # For You is typically the default timeline
            current_url = self.page.url
            if '/home' not in current_url:
                await self.page.goto('https://x.com/home', wait_until='domcontentloaded', timeout=30000)
                await asyncio.sleep(5)
            
            # For You is usually already selected, but let's try to click it to be sure
            for_you_selectors = [
                '[role="tablist"] [role="tab"]:has-text("For you")',
                '[role="tablist"] [role="tab"]:has-text("为你推荐")',
                '[role="tab"]:has-text("For you")',
                '[role="tab"]:has-text("为你推荐")'
            ]
            
            for selector in for_you_selectors:
                try:
                    element = self.page.locator(selector)
                    if await element.count() > 0:
                        await element.first.click()
                        self.logger.info("For You tab clicked")
                        await asyncio.sleep(3)
                        break
                except:
                    continue
            
            self.logger.info("For You timeline ready")
            return True
            
        except Exception as e:
            self.logger.warning(f"For You timeline navigation error: {e}")
            return True  # Continue anyway as it's usually the default
