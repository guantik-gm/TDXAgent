"""
TDXAgent - Personal Information AI Assistant

Main application entry point that orchestrates data collection,
processing, and report generation across multiple platforms.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import logging
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.config_manager import ConfigManager
from utils.logger import TDXLogger
from storage.jsonl_storage import JSONLStorage
from storage.media_storage import MediaStorage
from llm.openai_provider import OpenAIProvider
from llm.gemini_provider import GeminiProvider
from llm.claude_provider import ClaudeProvider
from llm.gemini_cli_provider import GeminiCliProvider
from processors.batch_processor import BatchProcessor, BatchConfig
from processors.report_generator import ReportGenerator
from processors.prompt_manager import PromptManager
from utils.link_generator import LinkGenerator
from scrapers.telegram_scraper import TelegramScraper
from scrapers.discord_scraper import DiscordScraper
from scrapers.twitter_scraper import TwitterScraper
from scrapers.gmail_scraper import GmailScraper


class TDXAgent:
    """
    Main TDXAgent application orchestrator.
    
    Coordinates data collection, processing, and reporting across
    Twitter/X, Telegram, and Discord platforms.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize TDXAgent.
        
        Args:
            config_path: Path to configuration file
        """
        self.console = Console()
        self.config_manager = ConfigManager(config_path)
        
        # Initialize logging
        self.logger = TDXLogger.setup_application_logging(
            self.config_manager.app.data_directory,
            self.config_manager.app.log_level
        )
        
        # Initialize components  
        self.storage = JSONLStorage(self.config_manager.app.data_directory)
        self.media_storage = MediaStorage(self.config_manager.app.data_directory)
        
        # Get reports directory from config (fallback to default)
        reports_dir = self.config_manager.output.get('reports_directory', f"{self.config_manager.app.data_directory}/reports")
        self.report_generator = ReportGenerator(reports_dir)
        
        # Track execution start time for reports (UTC+8)
        from datetime import timedelta
        utc_plus_8 = timezone(timedelta(hours=8))
        self.execution_start_time = datetime.now(utc_plus_8)
        
        # Initialize LLM provider
        self.llm_provider = self._initialize_llm_provider()
        
        # Initialize prompt manager and link generator
        self.prompt_manager = PromptManager()
        self.link_generator = LinkGenerator()
        
        self.batch_processor = BatchProcessor(
            self.llm_provider, 
            llm_config=self.config_manager.llm
        ) if self.llm_provider else None
        
        # Initialize scrapers
        self.scrapers = self._initialize_scrapers()
        
        self.logger.info("TDXAgent initialized successfully")
    
    def _initialize_llm_provider(self):
        """Initialize LLM provider based on configuration."""
        try:
            provider_name = self.config_manager.llm.provider
            
            if provider_name == 'openai':
                # Merge provider-specific config with general LLM config
                openai_config = {
                    **self.config_manager.llm.openai,
                    'max_requests_per_minute': self.config_manager.llm.max_requests_per_minute,
                    'max_tokens_per_minute': self.config_manager.llm.max_tokens_per_minute,
                    'timeout': self.config_manager.llm.timeout,
                    'max_retries': self.config_manager.llm.max_retries,
                    'retry_delay': self.config_manager.llm.retry_delay,
                    'max_tokens': self.config_manager.llm.max_tokens
                }
                return OpenAIProvider(openai_config, data_directory=str(self.config_manager.get_data_directory()))
            elif provider_name == 'gemini':
                # Merge provider-specific config with general LLM config
                gemini_config = {
                    **self.config_manager.llm.gemini,
                    'max_requests_per_minute': self.config_manager.llm.max_requests_per_minute,
                    'max_tokens_per_minute': self.config_manager.llm.max_tokens_per_minute,
                    'timeout': self.config_manager.llm.timeout,
                    'max_retries': self.config_manager.llm.max_retries,
                    'retry_delay': self.config_manager.llm.retry_delay,
                    'max_tokens': self.config_manager.llm.max_tokens
                }
                return GeminiProvider(gemini_config, data_directory=str(self.config_manager.get_data_directory()))
            elif provider_name == 'claude_cli':
                # Merge provider-specific config with general LLM config
                claude_config = {
                    **self.config_manager.llm.claude_cli,
                    'max_requests_per_minute': self.config_manager.llm.max_requests_per_minute,
                    'max_tokens_per_minute': self.config_manager.llm.max_tokens_per_minute,
                    'timeout': self.config_manager.llm.claude_cli.get('timeout', 120),
                    'max_retries': self.config_manager.llm.max_retries,
                    'retry_delay': self.config_manager.llm.retry_delay,
                    'max_tokens': self.config_manager.llm.max_tokens,
                    'enable_prompt_files': getattr(self.config_manager.llm, 'enable_prompt_files', True)
                }
                return ClaudeProvider(claude_config, data_directory=str(self.config_manager.get_data_directory()))
            elif provider_name == 'gemini_cli':
                # Merge provider-specific config with general LLM config
                gemini_cli_config = {
                    **self.config_manager.llm.gemini_cli,
                    'max_requests_per_minute': self.config_manager.llm.max_requests_per_minute,
                    'max_tokens_per_minute': self.config_manager.llm.max_tokens_per_minute,
                    'timeout': self.config_manager.llm.gemini_cli.get('timeout', 120),
                    'max_retries': self.config_manager.llm.max_retries,
                    'retry_delay': self.config_manager.llm.retry_delay,
                    'max_tokens': self.config_manager.llm.max_tokens,
                    'enable_prompt_files': getattr(self.config_manager.llm, 'enable_prompt_files', True)
                }
                return GeminiCliProvider(gemini_cli_config, data_directory=str(self.config_manager.get_data_directory()))
            else:
                self.logger.error(f"Unknown LLM provider: {provider_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider: {e}")
            return None
    
    def _initialize_scrapers(self) -> Dict[str, Any]:
        """Initialize platform scrapers."""
        scrapers = {}
        
        try:
            if self.config_manager.telegram.enabled:
                scrapers['telegram'] = TelegramScraper(
                    self.config_manager.telegram.__dict__, 
                    self.config_manager.app.data_directory
                )
            
            if self.config_manager.discord.enabled:
                scrapers['discord'] = DiscordScraper(self.config_manager.discord.__dict__)
            
            if self.config_manager.twitter.enabled:
                scrapers['twitter'] = TwitterScraper(self.config_manager.twitter.__dict__)
            
            if hasattr(self.config_manager, 'gmail') and self.config_manager.gmail.enabled:
                scrapers['gmail'] = GmailScraper(self.config_manager.gmail.__dict__)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize scrapers: {e}")
        
        return scrapers
    
    async def run_collection(self, hours_back: int = None, platforms: List[str] = None) -> Dict[str, Any]:
        """
        Run data collection from specified platforms.
        
        Args:
            hours_back: Hours back to collect data (uses config default if None)
            platforms: List of platforms to collect from (all enabled if None)
            
        Returns:
            Dictionary of collection results
        """
        hours_back = hours_back or self.config_manager.app.default_hours_to_fetch
        platforms = platforms or list(self.scrapers.keys())
        
        results = {}
        
        for platform in platforms:
            if platform not in self.scrapers:
                self.logger.warning(f"Platform {platform} not available")
                continue
            
            try:
                scraper = self.scrapers[platform]
                
                # Authenticate without progress bar interference
                self.console.print(f"🔑 正在认证 {platform}...")
                auth_success = await scraper.authenticate()
                
                if not auth_success:
                    self.logger.error(f"Failed to authenticate {platform}")
                    results[platform] = {'error': 'Authentication failed'}
                    continue
                
                # Now show progress for data collection
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task(f"收集 {platform} 数据...", total=None)
                    
                    # Scrape data
                    scraping_result = await scraper.scrape_with_monitoring(hours_back)
                    
                    # Store data
                    if scraping_result.messages:
                        success_count, failed_count = await self.storage.store_messages(
                            platform, scraping_result.messages
                        )
                        
                        self.logger.info(
                            f"Stored {success_count} {platform} messages "
                            f"({failed_count} failed)"
                        )
                    
                    results[platform] = {
                        'scraping_result': scraping_result,
                        'stored_messages': len(scraping_result.messages)
                    }
                    
                    # Cleanup
                    await scraper.cleanup()
                    
            except Exception as e:
                self.logger.error(f"Failed to collect {platform} data: {e}")
                results[platform] = {'error': str(e)}
        
        return results
    
    async def run_analysis(self, platforms: List[str] = None, hours_back: int = None) -> Dict[str, Any]:
        """
        Run AI analysis on collected data.
        
        Args:
            platforms: Platforms to analyze (all if None)
            hours_back: Hours back to analyze (uses config default if None)
            
        Returns:
            Dictionary of analysis results
        """
        if not self.batch_processor:
            self.logger.error("No LLM provider available for analysis")
            return {}
        
        hours_back = hours_back or self.config_manager.app.default_hours_to_fetch
        platforms = platforms or list(self.scrapers.keys())
        
        # Calculate time range for analysis (more precise than date-based)
        from datetime import datetime, timedelta, timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)
        
        # Also calculate date range for logging
        start_date = start_time.date()
        end_date = end_time.date()
        
        self.logger.info(f"分析时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 到 {end_time.strftime('%Y-%m-%d %H:%M')} ({hours_back}小时)")
        
        # 🎯 按平台独立分析 - 单独处理每个平台的数据
        platform_results = {}
        total_messages = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Get pure investment analysis template (platform-agnostic)
            prompt_template = await self.prompt_manager.get_template('pure_investment_analysis')
            
            if not prompt_template:
                self.logger.error("Pure investment analysis template not found")
                return {'error': 'Pure investment analysis template not available'}
            
            # 按平台独立处理
            for platform in platforms:
                platform_enabled = self.config_manager.is_platform_enabled(platform)
                self.logger.info(f"平台 {platform} 启用状态: {platform_enabled}")
                
                if not platform_enabled:
                    self.logger.warning(f"平台 {platform} 被禁用，跳过分析")
                    continue
                
                platform_task = progress.add_task(f"分析 {platform} 平台数据...", total=None)
                
                try:
                    # 使用精确时间范围收集当前平台数据
                    messages = await self.storage.get_messages_by_time_range(
                        platform,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if not messages:
                        self.logger.info(f"No messages found for {platform}")
                        platform_results[platform] = {
                            'batch_result': None,
                            'total_messages_analyzed': 0,
                            'error': f'No data available for {platform}'
                        }
                        continue
                    
                    total_messages += len(messages)
                    self.logger.info(f"收集到 {len(messages)} 条 {platform} 消息")
                    
                    # 🤖 执行单平台分析 - 使用自动分批处理
                    batch_result = await self.batch_processor.process_messages_with_template(
                        messages=messages,
                        prompt_template=prompt_template,
                        platform=platform,
                        formatted_messages=""  # 让BatchProcessor内部处理分批和格式化
                    )
                    
                    platform_results[platform] = {
                        'batch_result': batch_result,
                        'total_messages_analyzed': len(messages),
                        'platform': platform
                    }
                    
                    self.logger.info(f"完成 {platform} 平台分析: {len(messages)} 条消息")
                    
                    # 平台间缓冲等待时间（除了最后一个平台）
                    current_platform_index = list(platforms).index(platform)
                    if current_platform_index < len(list(platforms)) - 1:  # 不是最后一个平台
                        delay = self.config_manager.llm.multi_platform_delay
                        self.logger.info(f"平台间等待 {delay} 秒缓冲时间...")
                        await asyncio.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze {platform} data: {e}")
                    platform_results[platform] = {
                        'batch_result': None,
                        'total_messages_analyzed': 0,
                        'error': str(e)
                    }
                
                progress.advance(platform_task)
            
            if not platform_results:
                self.logger.warning("No data collected from any platform")
                return {'error': 'No data available for analysis'}
            
            results = {
                'platform_analysis': platform_results,
                'total_messages_analyzed': total_messages,
                'platforms_included': list(platform_results.keys())
            }
            
            self.logger.info(
                f"完成按平台独立分析: {total_messages} 条消息来自 {len(platform_results)} 个平台"
            )
        
        return results
    
    async def _format_platform_data(self, messages: List[Dict[str, Any]], platform: str) -> str:
        """
        格式化单个平台数据为AI分析的输入格式 - 完全平台无关。
        
        Args:
            messages: 平台消息数据
            platform: 平台名称（保留参数用于兼容性，但不再使用）
            
        Returns:
            格式化后的统一数据字符串
        """
        if not messages:
            return ""
        
        try:
            # 使用新的统一格式化方法 - AI完全平台无关
            return self.link_generator.format_messages_unified(messages)
                
        except Exception as e:
            self.logger.error(f"Failed to format platform data: {e}")
            # Fallback to basic unified formatting
            formatted_lines = []
            for i, msg in enumerate(messages, 1):
                author = msg.get('author', {}).get('name', 'Unknown')
                content = msg.get('content', {}).get('text', '')
                timestamp = msg.get('metadata', {}).get('posted_at', '')[:16]
                platform_name = msg.get('platform', '').title()
                if content.strip():
                    formatted_lines.append(f"{i}. [{timestamp}] {author}@{platform_name}: {content}")
            return '\n'.join(formatted_lines)
    
    async def _format_cross_platform_data(self, all_platform_data: Dict[str, List]) -> str:
        """
        格式化跨平台数据为AI分析的统一输入格式 - 完全平台无关。
        
        Args:
            all_platform_data: 各平台的消息数据
            
        Returns:
            格式化后的统一数据字符串
        """
        # 合并所有平台的消息
        all_messages = []
        for platform, messages in all_platform_data.items():
            if messages:
                all_messages.extend(messages)
        
        # 使用统一格式化方法 - AI完全平台无关
        return self.link_generator.format_messages_unified(all_messages)
    
    def _get_platform_display_name(self, platform: str) -> str:
        """获取平台显示名称"""
        platform_names = {
            'twitter': '🐦 Twitter/X',
            'telegram': '✈️ Telegram', 
            'discord': '💬 Discord',
            'gmail': '📧 Gmail'
        }
        return platform_names.get(platform, f"📱 {platform.title()}")
    
    async def generate_reports(self, analysis_results: Dict[str, Any], hours_back: int = None) -> List[str]:
        """
        Generate unified comprehensive report from cross-platform analysis results.
        
        Args:
            analysis_results: Results from run_analysis (now contains cross_platform_analysis)
            hours_back: Hours back that were processed (for filename generation)
            
        Returns:
            List containing the unified report file path
        """
        report_paths = []
        
        try:
            # 检查是否有按平台分析结果
            if 'platform_analysis' not in analysis_results:
                self.logger.error("No platform analysis results found")
                return []
            
            platform_results = analysis_results['platform_analysis']
            
            
            # Generate individual platform reports
            self.logger.debug(f"开始生成 {len(platform_results)} 个平台的报告")
            for platform, platform_result in platform_results.items():
                if not platform_result.get('batch_result') or not platform_result['batch_result'].summaries:
                    self.logger.warning(f"No analysis summaries available for {platform}")
                    continue
                
                self.logger.debug(f"正在生成 {platform} 平台报告...")
                
                # Generate platform-specific report
                platform_path = await self.report_generator.generate_platform_report(
                    platform=platform,
                    batch_result=platform_result['batch_result'],
                    messages=platform_result.get('messages', []),
                    start_time=self.execution_start_time,
                    hours_back=hours_back
                )
                
                if platform_path:
                    report_paths.append(platform_path)
                    self.logger.debug(f"✅ {platform} 平台报告已生成: {Path(platform_path).name}")
                    # Note: ReportGenerator already logs the generation success
            
            # Generate consolidated report if multiple platforms have reports
            if len(report_paths) > 1:
                self.logger.info(f"生成多平台汇总报告，整合 {len(report_paths)} 个平台报告...")
                
                # 生成汇总报告：简单地用 --- 分割各平台报告内容
                consolidated_path = await self._generate_consolidated_report(report_paths, hours_back)
                
                if consolidated_path:
                    # 根据配置决定是否删除单独的平台报告文件
                    multi_platform_config = self.config_manager.output.get('multi_platform_reports', {})
                    keep_individual_reports = multi_platform_config.get('keep_individual_reports', False)
                    
                    if not keep_individual_reports:
                        # 删除单独的平台报告文件
                        for platform_path in report_paths:
                            try:
                                Path(platform_path).unlink()
                                self.logger.debug(f"已删除单独平台报告: {Path(platform_path).name}")
                            except Exception as e:
                                self.logger.warning(f"删除平台报告失败 {platform_path}: {e}")
                        
                        # 只保留汇总报告路径
                        report_paths = [consolidated_path]
                        self.logger.info(f"已生成汇总报告并清理单独报告: {Path(consolidated_path).name}")
                    else:
                        # 保留分平台报告，添加汇总报告到路径列表
                        report_paths.append(consolidated_path)
                        self.logger.info(f"已生成汇总报告并保留分平台报告: {Path(consolidated_path).name}")
            else:
                self.logger.info(f"跳过汇总报告生成: 只有 {len(report_paths)} 个平台报告")
            
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
        
        return report_paths
    
    async def _generate_consolidated_report(self, platform_report_paths: List[str], hours_back: int = None) -> str:
        """
        生成简单的汇总报告：将各平台报告用 --- 分割整合到一个文件
        
        Args:
            platform_report_paths: 各平台报告文件路径列表
            hours_back: 小时数（用于文件名）
            
        Returns:
            汇总报告文件路径
        """
        try:
            # 读取所有平台报告内容
            all_contents = []
            
            for report_path in platform_report_paths:
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        all_contents.append(content)
                except Exception as e:
                    self.logger.warning(f"读取报告失败 {report_path}: {e}")
                    continue
            
            if not all_contents:
                self.logger.error("没有可用的平台报告内容")
                return None
            
            # 用 --- 分割符整合所有报告
            consolidated_content = "\n\n---\n\n".join(all_contents)
            
            # 生成汇总报告文件名
            timestamp = datetime.now().strftime("%Y年%m月%d日_%H时%M分")
            
            if hours_back:
                if hours_back == 1:
                    period_desc = "1小时"
                elif hours_back == 12:
                    period_desc = "12小时"
                elif hours_back == 24:
                    period_desc = "1天"
                elif hours_back < 24:
                    period_desc = f"{hours_back}小时"
                elif hours_back % 24 == 0:
                    days = hours_back // 24
                    period_desc = f"{days}天"
                else:
                    period_desc = f"{hours_back}小时"
            else:
                period_desc = "多小时"
            
            filename = f"TDXAgent_多平台汇总报告_{timestamp}_{period_desc}.md"
            consolidated_path = self.report_generator.output_directory / filename
            
            # 写入汇总报告
            with open(consolidated_path, 'w', encoding='utf-8') as f:
                f.write(consolidated_content)
            
            return str(consolidated_path)
            
        except Exception as e:
            self.logger.error(f"生成汇总报告失败: {e}")
            return None
    
    async def run_full_pipeline(self, hours_back: int = None, platforms: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete TDXAgent pipeline: collect -> analyze -> report.
        
        Args:
            hours_back: Hours back to process
            platforms: Platforms to process
            
        Returns:
            Complete pipeline results
        """
        self.console.print(Panel.fit("🚀 启动 TDXAgent 完整流程", style="bold blue"))
        
        # Step 1: Data Collection
        self.console.print("\n📥 步骤 1: 数据收集")
        collection_results = await self.run_collection(hours_back, platforms)
        
        # Step 2: AI Analysis
        self.console.print("\n🤖 步骤 2: AI 分析")
        analysis_results = await self.run_analysis(platforms, hours_back)
        
        # Step 3: Report Generation
        self.console.print("\n📊 步骤 3: 生成报告")
        report_paths = await self.generate_reports(analysis_results, hours_back)
        
        # Display summary
        self._display_pipeline_summary(collection_results, analysis_results, report_paths)
        
        return {
            'collection_results': collection_results,
            'analysis_results': analysis_results,
            'report_paths': report_paths
        }
    
    def _display_pipeline_summary(self, collection_results: Dict[str, Any], 
                                 analysis_results: Dict[str, Any], 
                                 report_paths: List[str]) -> None:
        """Display pipeline execution summary."""
        
        # Create summary table
        table = Table(title="TDXAgent 执行摘要")
        table.add_column("平台", style="cyan")
        table.add_column("收集消息", style="green")
        table.add_column("分析消息", style="yellow")
        table.add_column("状态", style="magenta")
        
        for platform in set(list(collection_results.keys()) + list(analysis_results.keys())):
            # Collection info
            collection_info = collection_results.get(platform, {})
            collected = collection_info.get('stored_messages', 0)
            
            # Analysis info
            analysis_info = analysis_results.get(platform, {})
            if 'batch_result' in analysis_info:
                analyzed = analysis_info['batch_result'].processed_messages
                status = f"✅ {analysis_info['batch_result'].success_rate:.1f}%"
            else:
                analyzed = 0
                status = "❌ 失败"
            
            table.add_row(
                platform.title(),
                str(collected),
                str(analyzed),
                status
            )
        
        self.console.print(table)
        
        # Report info
        if report_paths:
            if len(report_paths) == 1:
                self.console.print(f"\n📄 生成综合报告:")
            else:
                self.console.print(f"\n📄 生成了 {len(report_paths)} 个报告:")
            
            for path in report_paths:
                self.console.print(f"  • {Path(path).name}")
        
        self.console.print(Panel.fit("✨ TDXAgent 流程执行完成!", style="bold green"))


# CLI Interface
@click.group()
@click.option('--config', default='config.yaml', help='配置文件路径')
@click.pass_context
def cli(ctx, config):
    """TDXAgent - 个人信息 AI 助理"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command()
@click.option('--hours', default=12, help='收集多少小时前的数据')
@click.option('--platforms', help='指定平台 (逗号分隔)')
@click.pass_context
def collect(ctx, hours, platforms):
    """收集社交媒体数据"""
    async def run():
        agent = TDXAgent(ctx.obj['config'])
        platform_list = platforms.split(',') if platforms else None
        results = await agent.run_collection(hours, platform_list)
        return results
    
    asyncio.run(run())


@cli.command()
@click.option('--hours', default=12, help='分析多少小时前的数据')
@click.option('--platforms', help='指定平台 (逗号分隔)')
@click.pass_context
def analyze(ctx, hours, platforms):
    """分析收集的数据并生成报告"""
    async def run():
        agent = TDXAgent(ctx.obj['config'])
        platform_list = platforms.split(',') if platforms else None
        
        # 运行分析
        analysis_results = await agent.run_analysis(platform_list, hours)
        
        # 生成报告
        report_paths = await agent.generate_reports(analysis_results, hours)
        
        # 显示结果
        if report_paths:
            agent.console.print(f"\n📄 生成了 {len(report_paths)} 个报告:")
            for path in report_paths:
                agent.console.print(f"  • {Path(path).name}")
        else:
            agent.console.print("❌ 没有生成报告")
        
        return {'analysis_results': analysis_results, 'report_paths': report_paths}
    
    asyncio.run(run())


@cli.command()
@click.option('--hours', default=12, help='处理多少小时前的数据')
@click.option('--platforms', help='指定平台 (逗号分隔)')
@click.pass_context
def run(ctx, hours, platforms):
    """运行完整的 TDXAgent 流程"""
    async def run_pipeline():
        agent = TDXAgent(ctx.obj['config'])
        platform_list = platforms.split(',') if platforms else None
        results = await agent.run_full_pipeline(hours, platform_list)
        return results
    
    asyncio.run(run_pipeline())


@cli.command()
@click.pass_context
def status(ctx):
    """显示 TDXAgent 状态"""
    async def show_status():
        agent = TDXAgent(ctx.obj['config'])
        
        # Show configuration status
        console = Console()
        console.print("📋 TDXAgent 状态", style="bold blue")
        
        # Platform status
        table = Table(title="平台配置")
        table.add_column("平台", style="cyan")
        table.add_column("状态", style="green")
        
        platforms = ['twitter', 'telegram', 'discord']
        for platform in platforms:
            enabled = agent.config_manager.is_platform_enabled(platform)
            status = "✅ 启用" if enabled else "❌ 禁用"
            table.add_row(platform.title(), status)
        
        console.print(table)
        
        # Storage stats
        stats = await agent.storage.get_storage_stats()
        console.print(f"\n💾 存储统计: {stats['total_files']} 文件")
    
    asyncio.run(show_status())


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
