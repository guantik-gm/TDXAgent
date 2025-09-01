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
        self.report_generator = ReportGenerator(reports_dir, config_manager=self.config_manager)
        
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
        Run unified multi-platform AI analysis on collected data.
        
        Args:
            platforms: Platforms to analyze (all if None)
            hours_back: Hours back to analyze (uses config default if None)
            
        Returns:
            Dictionary with unified analysis results
        """
        if not self.batch_processor:
            self.logger.error("No LLM provider available for analysis")
            return {}
        
        hours_back = hours_back or self.config_manager.app.default_hours_to_fetch
        platforms = platforms or list(self.scrapers.keys())
        
        # Calculate time range for analysis (使用本地时区)
        from datetime import datetime, timedelta
        end_time = datetime.now()  # 本地时区时间
        start_time = end_time - timedelta(hours=hours_back)
        
        self.logger.info(f"分析时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 到 {end_time.strftime('%Y-%m-%d %H:%M')} ({hours_back}小时，本地时间)")
        
        # 🎯 统一多平台分析 - 收集所有平台数据后统一分析
        all_platform_data = {}
        data_file_paths = {}  # 记录各平台使用的数据文件绝对路径
        total_messages = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Step 1: 收集所有平台数据
            data_collection_task = progress.add_task("收集所有平台数据...", total=None)
            
            for platform in platforms:
                platform_enabled = self.config_manager.is_platform_enabled(platform)
                self.logger.info(f"平台 {platform} 启用状态: {platform_enabled}")
                
                if not platform_enabled:
                    self.logger.warning(f"平台 {platform} 被禁用，跳过收集")
                    continue
                
                try:
                    # 使用精确时间范围收集当前平台数据
                    messages = await self.storage.get_messages_by_time_range(
                        platform,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if messages:
                        all_platform_data[platform] = messages
                        total_messages += len(messages)
                        
                        # 获取实际使用的数据文件路径
                        # 由于可能跨多个日期，我们选择最新的文件路径作为代表
                        latest_date = end_time.date()
                        data_file_path = self.storage._get_file_path(platform, latest_date)
                        data_file_paths[platform] = str(data_file_path.absolute())
                        
                        self.logger.info(f"收集到 {len(messages)} 条 {platform} 消息")
                    else:
                        self.logger.info(f"No messages found for {platform}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to collect {platform} data: {e}")
            
            progress.advance(data_collection_task)
            
            if not all_platform_data:
                self.logger.warning("No data collected from any platform")
                return {'error': 'No data available for analysis'}
            
            # Step 2: 获取纯投资分析模板 (完全平台无关)
            prompt_template = await self.prompt_manager.get_template('pure_investment_analysis')
            if not prompt_template:
                self.logger.error("Pure investment analysis template not found")
                return {'error': 'Pure investment analysis template not available'}
            
            # Step 3: 统一多平台分析
            analysis_task = progress.add_task("执行统一多平台分析...", total=None)
            
            try:
                # 🤖 执行统一多平台分析 - 使用新的统一处理方法
                unified_result = await self.batch_processor.process_unified_multi_platform_messages(
                    all_platform_data=all_platform_data,
                    prompt_template=prompt_template
                )
                
                self.logger.info(
                    f"完成统一多平台分析: {total_messages} 条消息来自 {len(all_platform_data)} 个平台"
                )
                
                # 🎯 构建独立分析+整合结果结构 - 只显示成功分析的平台
                # 从BatchResult中获取成功分析的平台信息
                successful_platforms = getattr(unified_result, 'successful_platforms', [])
                
                # 如果没有成功平台信息，fallback到所有有数据的平台
                if not successful_platforms:
                    successful_platforms = [p for p, msgs in all_platform_data.items() if len(msgs) > 0]
                
                results = {
                    'unified_analysis': unified_result,
                    'total_messages_analyzed': total_messages,
                    'platforms_included': successful_platforms,  # 只显示参与成功分析的平台
                    'platforms_attempted': list(all_platform_data.keys()),  # 所有尝试的平台
                    'platform_message_counts': {
                        platform: len(messages) 
                        for platform, messages in all_platform_data.items()
                        if platform in successful_platforms
                    },
                    'data_file_paths': {
                        platform: path 
                        for platform, path in data_file_paths.items()
                        if platform in successful_platforms
                    }
                }
                
                progress.advance(analysis_task)
                return results
                
            except Exception as e:
                self.logger.error(f"Failed to run unified analysis: {e}")
                return {
                    'error': f'Unified analysis failed: {str(e)}',
                    'platforms_included': list(all_platform_data.keys()),
                    'total_messages_analyzed': total_messages
                }
        
        return {
            'error': 'Analysis workflow interrupted',
            'platforms_included': [],
            'total_messages_analyzed': 0
        }
    
    
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
        Generate unified report from multi-platform analysis results.
        
        Args:
            analysis_results: Results from run_analysis (contains unified_analysis)
            hours_back: Hours back that were processed (for filename generation)
            
        Returns:
            List containing the unified report file path
        """
        report_paths = []
        
        try:
            # 检查是否有分析错误
            if 'error' in analysis_results:
                self.logger.error(f"Analysis failed: {analysis_results['error']}")
                # 生成错误报告
                error_path = await self.report_generator.generate_error_report(
                    error_message=analysis_results['error'],
                    platforms_attempted=analysis_results.get('platforms_included', []),
                    total_messages=analysis_results.get('total_messages_analyzed', 0),
                    start_time=self.execution_start_time,
                    hours_back=hours_back
                )
                if error_path:
                    report_paths.append(error_path)
                return report_paths
            
            # 检查是否有统一分析结果
            if 'unified_analysis' not in analysis_results:
                self.logger.error("No unified analysis results found")
                # 生成错误报告
                error_path = await self.report_generator.generate_error_report(
                    error_message="分析流程未能产生有效结果，可能是系统内部错误",
                    platforms_attempted=analysis_results.get('platforms_included', []),
                    total_messages=analysis_results.get('total_messages_analyzed', 0),
                    start_time=self.execution_start_time,
                    hours_back=hours_back
                )
                if error_path:
                    report_paths.append(error_path)
                return report_paths
            
            unified_result = analysis_results['unified_analysis']
            platforms_included = analysis_results.get('platforms_included', [])
            total_messages = analysis_results.get('total_messages_analyzed', 0)
            
            # 检查统一分析结果是否有效
            if not unified_result or not unified_result.summaries:
                self.logger.warning("No analysis summaries available in unified result")
                # 生成错误报告，但包含更多调试信息
                error_details = []
                if not unified_result:
                    error_details.append("分析处理器未返回结果对象")
                elif not unified_result.summaries:
                    error_details.append("分析处理完成，但未产生分析摘要")
                
                error_path = await self.report_generator.generate_error_report(
                    error_message=f"分析失败: {'; '.join(error_details)}",
                    platforms_attempted=platforms_included,
                    total_messages=total_messages,
                    start_time=self.execution_start_time,
                    hours_back=hours_back,
                    analysis_result=unified_result  # 传递分析结果用于调试
                )
                if error_path:
                    report_paths.append(error_path)
                return report_paths
            
            self.logger.info(f"开始生成统一多平台分析报告...")
            self.logger.info(f"分析数据：{total_messages} 条消息来自 {len(platforms_included)} 个平台: {', '.join(platforms_included)}")
            
            # 生成统一多平台报告
            unified_path = await self.report_generator.generate_unified_report(
                batch_result=unified_result,
                platforms_included=platforms_included,
                total_messages=total_messages,
                start_time=self.execution_start_time,
                hours_back=hours_back,
                data_file_paths=analysis_results.get('data_file_paths', {})
            )
            
            if unified_path:
                report_paths.append(unified_path)
                self.logger.info(f"✅ 统一多平台分析报告已生成: {Path(unified_path).name}")
            else:
                self.logger.error("统一报告生成失败")
            
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
        
        return report_paths
    
    
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
        table.add_column("分析状态", style="yellow")
        
        # 处理新的统一分析结构
        platforms_included = analysis_results.get('platforms_included', [])
        platform_message_counts = analysis_results.get('platform_message_counts', {})
        unified_analysis = analysis_results.get('unified_analysis')
        
        # 显示各平台信息
        all_platforms = set(list(collection_results.keys()) + platforms_included)
        
        for platform in sorted(all_platforms):
            # Collection info
            collection_info = collection_results.get(platform, {})
            collected = collection_info.get('stored_messages', 0)
            
            # Analysis info - 新的统一结构
            analyzed_count = platform_message_counts.get(platform, 0)
            
            # 分析状态
            if platform in platforms_included and unified_analysis:
                if analyzed_count > 0:
                    status = f"✅ 已统一分析"
                else:
                    status = "⚠️ 无数据"
            elif platform in collection_results:
                status = "⏭️ 未分析"
            else:
                status = "❌ 失败"
            
            table.add_row(
                platform.title(),
                str(collected),
                status
            )
        
        self.console.print(table)
        
        # 统一分析摘要
        if unified_analysis:
            total_messages = analysis_results.get('total_messages_analyzed', 0)
            success_rate = unified_analysis.success_rate if hasattr(unified_analysis, 'success_rate') else 100
            self.console.print(f"\n🤖 统一多平台分析: {total_messages} 条消息，成功率 {success_rate:.1f}%")
            self.console.print(f"   涉及平台: {', '.join(platforms_included)}")
        
        # Report info
        if report_paths:
            if len(report_paths) == 1:
                self.console.print(f"\n📄 生成统一报告:")
            else:
                self.console.print(f"\n📄 生成了 {len(report_paths)} 个报告:")
            
            for path in report_paths:
                self.console.print(f"  • {Path(path).name}")
        
        self.console.print(Panel.fit("✨ TDXAgent 流程执行完成!", style="bold green"))
    
    
    def _log_task_summary(self, task_type: str, **kwargs) -> None:
        """
        Log task execution summary with grep-friendly format.
        
        Args:
            task_type: COLLECT or ANALYZE
            **kwargs: Task-specific parameters
        """
        # Create a standardized log entry format
        summary_parts = [f"[TASK_SUMMARY] {task_type}"]
        
        # Add key-value pairs
        for key, value in kwargs.items():
            if value is not None:
                summary_parts.append(f"{key}={value}")
        
        summary_line = " | ".join(summary_parts)
        
        # Log at INFO level to ensure visibility
        self.logger.info(summary_line)


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
        import time
        start_time = time.time()
        
        agent = TDXAgent(ctx.obj['config'])
        platform_list = platforms.split(',') if platforms else None
        results = await agent.run_collection(hours, platform_list)
        
        # Calculate execution time
        duration = int(time.time() - start_time)
        
        # Calculate summary statistics
        total_messages = 0
        platform_breakdown = []
        error_count = 0
        successful_platforms = []
        failed_platforms = []
        auth_failed_platforms = []
        
        for platform, result in results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    error_count += 1
                    failed_platforms.append(platform)
                    # Check for authentication errors
                    error_msg = result.get('error', '').lower()
                    if 'authentication failed' in error_msg or 'authenticate' in error_msg or 'auth' in error_msg:
                        auth_failed_platforms.append(platform)
                elif 'stored_messages' in result:
                    messages_count = result['stored_messages']
                    total_messages += messages_count
                    platform_breakdown.append(f"{platform}:{messages_count}")
                    successful_platforms.append(platform)
        
        # Determine overall status
        status = "SUCCESS" if error_count == 0 else "PARTIAL" if successful_platforms else "FAILED"
        
        # Log task summary with enhanced error information
        summary_params = {
            "platforms": ",".join(platform_list) if platform_list else "all",
            "hours": hours,
            "total_messages": total_messages,
            "platform_breakdown": ",".join(platform_breakdown) if platform_breakdown else "none",
            "status": status,
            "duration": f"{duration}s",
            "errors": error_count
        }
        
        # Add failed platform information
        if failed_platforms:
            summary_params["failed_platforms"] = ",".join(failed_platforms)
        
        # Add authentication failure information
        if auth_failed_platforms:
            summary_params["auth_failed"] = ",".join(auth_failed_platforms)
        
        agent._log_task_summary("COLLECT", **summary_params)
        
        # Display authentication warnings to user
        if auth_failed_platforms:
            agent.console.print(f"[bold red]⚠️  认证失败平台: {', '.join(auth_failed_platforms)}[/bold red]")
            agent.console.print("[yellow]请运行相应平台的认证命令重新进行认证[/yellow]")
        
        return results
    
    asyncio.run(run())


@cli.command()
@click.option('--hours', default=12, help='分析多少小时前的数据')
@click.option('--platforms', help='指定平台 (逗号分隔)')
@click.pass_context
def analyze(ctx, hours, platforms):
    """分析收集的数据并生成报告"""
    async def run():
        import time
        start_time = time.time()
        
        agent = TDXAgent(ctx.obj['config'])
        platform_list = platforms.split(',') if platforms else None
        
        # 运行分析
        analysis_results = await agent.run_analysis(platform_list, hours)
        
        # 生成报告
        report_paths = await agent.generate_reports(analysis_results, hours)
        
        # Calculate execution time
        duration = int(time.time() - start_time)
        
        # Calculate summary statistics
        total_messages = 0
        platform_breakdown = []
        error_count = 0
        successful_platforms = []
        failed_platforms = []
        auth_failed_platforms = []
        total_batches = 0
        
        if 'error' in analysis_results:
            error_count = 1
            status = "FAILED"
        elif 'unified_analysis' in analysis_results:
            # 🎯 统一分析架构 - 提取分析结果数据
            unified_result = analysis_results.get('unified_analysis')
            total_messages = analysis_results.get('total_messages_analyzed', 0)
            platforms_included = analysis_results.get('platforms_included', [])
            platform_counts = analysis_results.get('platform_message_counts', {})
            
            # 构建平台分解信息
            for platform, count in platform_counts.items():
                platform_breakdown.append(f"{platform}:{count}")
                successful_platforms.append(platform)
            
            # 检查统一分析结果是否成功
            if unified_result:
                # BatchResult成功判断：有成功的批次且有分析摘要
                has_successful_batches = unified_result.successful_batches > 0
                has_summaries = unified_result.summaries and len(unified_result.summaries) > 0
                has_no_critical_errors = unified_result.failed_batches == 0 or unified_result.successful_batches > 0
                
                if has_successful_batches and has_summaries and has_no_critical_errors:
                    status = "SUCCESS"
                    # 🎯 修复：使用正确的批次统计方式，与BatchProcessor保持一致
                    total_batches = unified_result.successful_batches + unified_result.failed_batches
                else:
                    error_count = 1
                    status = "FAILED"
                    # 不应该将成功的平台标记为失败
                    # if platforms_included:
                    #     failed_platforms.extend(platforms_included)
            else:
                error_count = 1
                status = "FAILED"
                if platforms_included:
                    failed_platforms.extend(platforms_included)
        else:
            status = "FAILED"
        
        # 显示结果
        if report_paths:
            agent.console.print(f"\n📄 生成了 {len(report_paths)} 个报告:")
            for path in report_paths:
                agent.console.print(f"  • {Path(path).name}")
        else:
            agent.console.print("❌ 没有生成报告")
        
        # Log task summary with enhanced error information
        summary_params = {
            "platforms": ",".join(platform_list) if platform_list else "all",
            "hours": hours,
            "total_messages": total_messages,
            "platform_breakdown": ",".join(platform_breakdown) if platform_breakdown else "none",
            "batches": total_batches if total_batches > 0 else "unknown",
            "reports": len(report_paths) if report_paths else 0,
            "status": status,
            "duration": f"{duration}s",
            "errors": error_count
        }
        
        # Add failed platform information
        if failed_platforms:
            summary_params["failed_platforms"] = ",".join(failed_platforms)
        
        # Add authentication failure information
        if auth_failed_platforms:
            summary_params["auth_failed"] = ",".join(auth_failed_platforms)
        
        agent._log_task_summary("ANALYZE", **summary_params)
        
        # Display authentication warnings to user
        if auth_failed_platforms:
            agent.console.print(f"[bold red]⚠️  认证失败平台: {', '.join(auth_failed_platforms)}[/bold red]")
            agent.console.print("[yellow]请检查相关平台的认证配置，可能需要重新认证[/yellow]")
        
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
