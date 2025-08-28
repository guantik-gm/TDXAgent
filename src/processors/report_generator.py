"""
Report generator for TDXAgent.

This module generates comprehensive markdown reports from processed
social media data and LLM analysis results.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from utils.logger import TDXLogger
from utils.helpers import ensure_directory_async, format_timestamp, sanitize_filename
from processors.batch_processor import BatchResult


class ReportGenerator:
    """
    Comprehensive report generator for TDXAgent.
    
    Features:
    - Markdown report generation
    - Platform-specific sections
    - Summary statistics
    - Media file references
    - Customizable templates
    - Multi-language support
    """
    
    def __init__(self, output_directory: str = "TDXAgent_Data/reports", start_time: Optional[datetime] = None, hours_back: Optional[int] = None, config_manager=None):
        """
        Initialize report generator.
        
        Args:
            output_directory: Directory to save reports
            start_time: Program start time for filename generation
            hours_back: Hours of data collected for filename generation
            config_manager: Configuration manager instance (optional)
        """
        self.output_directory = Path(output_directory)
        # Use UTC+8 timezone for start_time
        from datetime import timedelta
        utc_plus_8 = timezone(timedelta(hours=8))
        self.start_time = start_time or datetime.now(utc_plus_8)
        self.hours_back = hours_back
        self.logger = TDXLogger.get_logger("tdxagent.processors.reports")
        
        # Store config manager for accessing configuration
        self.config_manager = config_manager
        
        # Report templates
        self.templates = {
            'summary': self._get_summary_template(),
            'detailed': self._get_detailed_template(),
            'platform_specific': self._get_platform_template()
        }
        
        self.logger.info(f"Initialized report generator: {self.output_directory}")
    
    
    def _generate_filename(self, prefix: str) -> str:
        """
        Generate human-readable filename.
        
        Args:
            prefix: File prefix (e.g., "TDXAgent综合报告", "Twitter报告")
            
        Returns:
            Generated filename with human-readable format
        """
        # Format date for human readability
        date_str = self.start_time.strftime("%Y年%m月%d日")
        time_str = self.start_time.strftime("%H时%M分")
        
        # Generate hours description
        if self.hours_back:
            if self.hours_back == 1:
                duration_str = "1小时"
            elif self.hours_back == 24:
                duration_str = "1天"  
            elif self.hours_back == 168:  # 7*24
                duration_str = "1周"
            elif self.hours_back < 24:
                duration_str = f"{self.hours_back}小时"
            elif self.hours_back % 24 == 0:
                days = self.hours_back // 24
                duration_str = f"{days}天"
            else:
                duration_str = f"{self.hours_back}小时"
        else:
            duration_str = "未知时长"
        
        # Generate human-readable filename
        filename = f"{prefix}_{date_str}_{time_str}_{duration_str}.md"
        return filename
    
    def _write_file(self, file_path: Path, content: str) -> None:
        """Write content to file synchronously."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    async def generate_unified_report(self, 
                                    batch_result: BatchResult,
                                    platforms_included: List[str],
                                    total_messages: int,
                                    start_time: Optional[datetime] = None,
                                    hours_back: Optional[int] = None,
                                    data_file_paths: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a unified multi-platform report from unified analysis results.
        
        Args:
            batch_result: Unified analysis batch processing result
            platforms_included: List of platforms included in analysis
            total_messages: Total number of messages analyzed
            start_time: Program start time (overrides instance value if provided)
            hours_back: Hours of data collected (overrides instance value if provided)
            data_file_paths: Dict mapping platform to absolute data file path
            
        Returns:
            Path to generated unified report file
        """
        try:
            # Update timing parameters if provided
            if start_time:
                self.start_time = start_time
            if hours_back is not None:
                self.hours_back = hours_back
                
            await ensure_directory_async(self.output_directory)
            
            # Generate unified report content
            report_content = self._generate_unified_content(batch_result, platforms_included, total_messages)
            
            # Create filename with unified format
            filename = self._generate_filename("TDXAgent分析报告")
            
            # Create date-based subdirectory
            report_date = datetime.now()
            date_folder = report_date.strftime("%Y%m%d")
            date_directory = self.output_directory / date_folder
            await ensure_directory_async(date_directory)
            
            report_path = date_directory / filename
            
            # Write report
            await asyncio.to_thread(self._write_file, report_path, report_content)
            
            self.logger.info(f"Generated unified multi-platform report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate unified report: {e}")
            raise
    
    async def generate_error_report(self, 
                                  error_message: str,
                                  platforms_attempted: List[str],
                                  total_messages: int,
                                  start_time: Optional[datetime] = None,
                                  hours_back: Optional[int] = None,
                                  analysis_result: Optional = None) -> str:
        """
        Generate an error report when analysis fails.
        
        Args:
            error_message: The error that occurred
            platforms_attempted: List of platforms that were attempted
            total_messages: Total number of messages that were collected
            start_time: Program start time (overrides instance value if provided)
            hours_back: Hours of data collected (overrides instance value if provided)
            analysis_result: Any partial analysis result for debugging (optional)
            
        Returns:
            Path to generated error report file
        """
        try:
            # Update timing parameters if provided
            if start_time:
                self.start_time = start_time
            if hours_back is not None:
                self.hours_back = hours_back
                
            await ensure_directory_async(self.output_directory)
            
            # Generate error report content
            report_content = self._generate_error_content(
                error_message, platforms_attempted, total_messages, analysis_result
            )
            
            # Create filename with error format
            filename = self._generate_filename("分析错误报告")
            
            # Create date-based subdirectory
            report_date = datetime.now()
            date_folder = report_date.strftime("%Y%m%d")
            date_directory = self.output_directory / date_folder
            await ensure_directory_async(date_directory)
            
            report_path = date_directory / filename
            
            # Write report
            await asyncio.to_thread(self._write_file, report_path, report_content)
            
            self.logger.info(f"Generated error report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate error report: {e}")
            # Don't raise here - we don't want to fail the error report generation
            return None
    
    def _generate_unified_content(self, 
                                 batch_result: BatchResult,
                                 platforms_included: List[str],
                                 total_messages: int) -> str:
        """Generate unified multi-platform report content."""
        
        report_time = format_timestamp(format_type="human")
        
        # Get platform display names
        platform_names = [self._get_platform_display_name(p) for p in platforms_included]
        platforms_display = "、".join(platform_names)
        
        # Generate data files section
        if data_file_paths:
            data_files_lines = []
            for platform in platforms_included:
                if platform in data_file_paths:
                    platform_name = self._get_platform_display_name(platform)
                    file_path = data_file_paths[platform]
                    data_files_lines.append(f"- **{platform_name}**: `{file_path}`")
                else:
                    platform_name = self._get_platform_display_name(platform)
                    data_files_lines.append(f"- **{platform_name}**: 数据文件路径未知")
            data_files_section = "\n".join(data_files_lines)
        else:
            data_files_section = "数据文件路径信息不可用"
        
        content = f"""
**生成时间**: {report_time}  
**数据范围**: 最近 {self.hours_back or 'N/A'} 小时  
**分析模式**: 统一多平台分析

## 📊 处理统计

- **涉及平台**: {platforms_display}
- **总消息数**: {total_messages:,}
- **成功处理**: {batch_result.processed_messages:,} ({batch_result.success_rate:.1f}%)
- **批次处理**: {batch_result.successful_batches} 成功 / {batch_result.failed_batches} 失败
- **Token 使用**: {batch_result.total_tokens_used:,} (平均 {batch_result.average_tokens_per_batch:.0f}/批次)
- **处理时间**: {batch_result.processing_time:.1f} 秒
- **估算成本**: ${batch_result.total_cost:.4f}

### 📂 原始数据文件路径
{data_files_section}

## 🤖 统一AI分析结果

"""
        
        # Add unified AI analysis results
        if batch_result.summaries:
            content += f"""### 统一多平台投资分析

"""
            # 显示统一分析摘要
            for summary in batch_result.summaries:
                content += f"{summary}\n\n"
        else:
            content += "暂无 AI 分析结果。\n\n"
        
        
        # Add simple footer
        content += f"""*报告由 TDXAgent 自动生成 - {report_time}*
"""
        
        return content
    
    def _get_platform_display_name(self, platform: str) -> str:
        """Get display name for platform."""
        names = {
            'twitter': '🐦 Twitter/X',
            'telegram': '✈️ Telegram',
            'discord': '💬 Discord',
            'gmail': '📧 Gmail'
        }
        return names.get(platform.lower(), platform.title())
    
    def _generate_error_content(self, 
                               error_message: str,
                               platforms_attempted: List[str],
                               total_messages: int,
                               analysis_result = None) -> str:
        """Generate error report content."""
        
        report_time = format_timestamp(format_type="human")
        
        # Get platform display names
        if platforms_attempted:
            platform_names = [self._get_platform_display_name(p) for p in platforms_attempted]
            platforms_display = "、".join(platform_names)
        else:
            platforms_display = "无"
        
        content = f"""# TDXAgent 分析错误报告

**生成时间**: {report_time}

## 执行信息
- **分析平台**: {platforms_display}
- **收集消息**: {total_messages:,} 条
- **数据状态**: ✅ 已收集完成

## 错误日志
```
{error_message}
```

## 处理结果
- **数据收集**: ✅ 成功 ({total_messages:,} 条消息已保存)
- **AI分析**: ❌ 失败
- **报告生成**: ❌ 未完成

## 重新分析
解决错误后运行: `python src/main.py analyze --hours {self.hours_back or 12}`

"""
        
        
        content += f"*{report_time} - TDXAgent*"
        
        return content
    
    def _get_summary_template(self) -> str:
        """Get summary report template."""
        return """# {title}

**生成时间**: {timestamp}

## 📊 总体统计

{overall_stats}

## 🌐 平台概览

{platform_sections}

## ⚠️ 错误报告

{error_summary}

---

*报告由 TDXAgent 自动生成*
"""
    
    def _get_detailed_template(self) -> str:
        """Get detailed report template."""
        return """# {platform} 详细分析报告

**生成时间**: {timestamp}

## 📈 处理统计

{processing_stats}

## 🤖 AI 分析结果

{ai_summaries}

## 📝 消息样本

{message_samples}

## ⚠️ 错误详情

{error_details}

---

*报告由 TDXAgent 自动生成*
"""
    
    def _get_platform_template(self) -> str:
        """Get platform-specific template."""
        return self._get_detailed_template()
    
    async def list_reports(self) -> List[Dict[str, Any]]:
        """
        List all generated reports.
        
        Returns:
            List of report information
        """
        reports = []
        
        try:
            if not self.output_directory.exists():
                return reports
            
            for report_file in self.output_directory.glob("*.md"):
                try:
                    stat = report_file.stat()
                    reports.append({
                        'filename': report_file.name,
                        'path': str(report_file),
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to get info for {report_file}: {e}")
            
            # Sort by creation time (newest first)
            reports.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list reports: {e}")
        
        return reports
    
    async def cleanup_old_reports(self, days_old: int = 30) -> int:
        """
        Clean up old report files.
        
        Args:
            days_old: Remove reports older than this many days
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        try:
            if not self.output_directory.exists():
                return 0
            
            for report_file in self.output_directory.glob("*.md"):
                try:
                    if report_file.stat().st_mtime < cutoff_time:
                        report_file.unlink()
                        removed_count += 1
                        self.logger.debug(f"Removed old report: {report_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {report_file}: {e}")
            
            self.logger.info(f"Cleaned up {removed_count} old reports")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup reports: {e}")
        
        return removed_count
    
    def _add_batch_execution_details(self, content: str, platform_results: Dict[str, Any]) -> str:
        """
        添加批次执行详情到报告内容中
        
        Args:
            content: 现有的报告内容
            platform_results: 平台处理结果字典
            
        Returns:
            更新后的报告内容
        """
        # 导入BatchProcessor以使用get_execution_summary方法
        from processors.batch_processor import BatchProcessor
        
        # 检查是否存在有问题的批次
        all_problematic_batches = []
        platform_summaries = {}
        
        for platform, result_data in platform_results.items():
            if isinstance(result_data, dict) and 'batch_result' in result_data:
                batch_result = result_data['batch_result']
                
                # 检查BatchResult是否有batch_details属性（新版本才有）
                if hasattr(batch_result, 'batch_details') and batch_result.batch_details:
                    problematic_batches = batch_result.problematic_batches
                    if problematic_batches:
                        all_problematic_batches.extend(problematic_batches)
                        # 直接调用BatchProcessor的静态方法或创建简化的summary
                        platform_summaries[platform] = self._create_execution_summary(batch_result)
        
        # 如果没有问题批次，返回原内容
        if not all_problematic_batches:
            return content
        
        # 添加批次执行详情章节
        execution_details = f"""

## 🔍 批次执行详情

检测到 {len(all_problematic_batches)} 个有问题的批次，详情如下：

"""
        
        for platform, summary in platform_summaries.items():
            platform_name = self._get_platform_display_name(platform)
            execution_details += f"### {platform_name}\n\n"
            
            # 基本统计信息
            execution_details += f"""**执行统计**:
- 总批次数: {summary['total_batches']}
- 成功批次: {summary['successful_batches']}
- 失败批次: {summary['failed_batches']}
- 问题批次: {summary['problematic_count']}

"""
            
            # 失败批次详情
            if summary['failed_details']:
                execution_details += "**🚨 失败批次详情**:\n\n"
                for detail in summary['failed_details']:
                    execution_details += f"""- **批次 {detail['batch_number']}**: 
  - LLM命令: `{detail['llm_command']}`
  - 消息数: {detail['message_count']} 条
  - 处理时间: {detail['processing_time']:.2f}s
  - 错误信息: {detail['error_message']}

"""
            
            # 无效结果批次详情
            if summary['empty_result_details']:
                execution_details += "**⚠️ 无有效结果批次**:\n\n"
                for detail in summary['empty_result_details']:
                    execution_details += f"""- **批次 {detail['batch_number']}**: 
  - LLM命令: `{detail['llm_command']}`
  - 消息数: {detail['message_count']} 条
  - Token使用: {detail['tokens_used']}
  - 处理时间: {detail['processing_time']:.2f}s
  - 响应预览: {detail['response_preview']}

"""
            
            execution_details += "---\n\n"
        
        # 添加调试建议
        execution_details += """**🛠️ 调试建议**:

1. **失败批次**: 检查网络连接、API配额或模型可用性
2. **无效结果批次**: 
   - 检查输入数据质量和相关性
   - 考虑调整提示词模板
   - 验证消息时间范围是否包含相关内容
3. **重新运行**: 可以尝试重新运行相同时间范围的分析

"""
        
        return content + execution_details
    
    def _has_execution_details_support(self, batch_result) -> bool:
        """
        检查BatchResult是否支持batch_details（向后兼容性检查）
        
        Args:
            batch_result: BatchResult实例
            
        Returns:
            是否支持详细执行信息
        """
        return hasattr(batch_result, 'batch_details') and batch_result.batch_details is not None
    
    def _create_execution_summary(self, batch_result) -> dict:
        """
        创建批次执行汇总信息（独立于BatchProcessor的实现）
        
        Args:
            batch_result: BatchResult实例
            
        Returns:
            包含执行汇总的字典
        """
        if not hasattr(batch_result, 'batch_details') or not batch_result.batch_details:
            return {}
        
        problematic_batches = batch_result.problematic_batches
        failed_batches = batch_result.failed_batch_details
        empty_batches = batch_result.empty_result_batches
        
        return {
            'platform': batch_result.platform,
            'total_batches': batch_result.successful_batches + batch_result.failed_batches,
            'successful_batches': batch_result.successful_batches,
            'failed_batches': batch_result.failed_batches,
            'problematic_count': len(problematic_batches),
            'failed_details': [
                {
                    'batch_number': detail.batch_number,
                    'llm_command': getattr(detail, 'llm_command', detail.command_type),
                    'message_count': detail.message_count,
                    'error_message': detail.error_message,
                    'processing_time': detail.processing_time
                }
                for detail in failed_batches
            ],
            'empty_result_details': [
                {
                    'batch_number': detail.batch_number,
                    'llm_command': getattr(detail, 'llm_command', detail.command_type),
                    'message_count': detail.message_count,
                    'tokens_used': detail.tokens_used,
                    'processing_time': detail.processing_time,
                    'response_preview': detail.response_content[:100] + "..." if len(detail.response_content) > 100 else detail.response_content
                }
                for detail in empty_batches
            ]
        }
    
