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
    
    def __init__(self, output_directory: str = "TDXAgent_Data/reports", start_time: Optional[datetime] = None, hours_back: Optional[int] = None):
        """
        Initialize report generator.
        
        Args:
            output_directory: Directory to save reports
            start_time: Program start time for filename generation
            hours_back: Hours of data collected for filename generation
        """
        self.output_directory = Path(output_directory)
        # Use UTC+8 timezone for start_time
        from datetime import timedelta
        utc_plus_8 = timezone(timedelta(hours=8))
        self.start_time = start_time or datetime.now(utc_plus_8)
        self.hours_back = hours_back
        self.logger = TDXLogger.get_logger("tdxagent.processors.reports")
        
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
    
    async def generate_summary_report(self, 
                                    platform_results: Dict[str, BatchResult],
                                    report_title: str = "TDXAgent 数据分析报告",
                                    start_time: Optional[datetime] = None,
                                    hours_back: Optional[int] = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            platform_results: Dictionary of platform -> BatchResult
            report_title: Title for the report
            start_time: Program start time (overrides instance value if provided)
            hours_back: Hours of data collected (overrides instance value if provided)
            
        Returns:
            Path to generated report file
        """
        try:
            # Update timing parameters if provided
            if start_time:
                self.start_time = start_time
            if hours_back is not None:
                self.hours_back = hours_back
                
            await ensure_directory_async(self.output_directory)
            
            # Generate report content
            report_content = await self._generate_summary_content(platform_results, report_title)
            
            # Create filename with new format
            filename = self._generate_filename("Summary_Report")
            report_path = self.output_directory / filename
            
            # Write report
            await asyncio.to_thread(self._write_file, report_path, report_content)
            
            self.logger.info(f"Generated summary report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            raise
    
    async def generate_platform_report(self, 
                                     platform: str,
                                     batch_result: BatchResult,
                                     messages: List[Dict[str, Any]],
                                     all_platform_results: Optional[Dict[str, BatchResult]] = None,
                                     start_time: Optional[datetime] = None,
                                     hours_back: Optional[int] = None) -> str:
        """
        Generate a platform-specific detailed report.
        
        Args:
            platform: Platform name
            batch_result: Batch processing result
            messages: Original messages
            all_platform_results: All platform results for summary section (optional)
            start_time: Program start time (overrides instance value if provided)
            hours_back: Hours of data collected (overrides instance value if provided)
            
        Returns:
            Path to generated report file
        """
        try:
            # Update timing parameters if provided
            if start_time:
                self.start_time = start_time
            if hours_back is not None:
                self.hours_back = hours_back
                
            await ensure_directory_async(self.output_directory)
            
            # Generate report content
            report_content = self._generate_platform_content(platform, batch_result, messages, all_platform_results)
            
            # Create filename with new format
            filename = self._generate_filename(f"{platform.title()}_Report")
            report_path = self.output_directory / filename
            
            # Write report
            await asyncio.to_thread(self._write_file, report_path, report_content)
            
            self.logger.info(f"Generated {platform} report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate {platform} report: {e}")
            raise
    
    async def _generate_summary_content(self, 
                                      platform_results: Dict[str, BatchResult],
                                      report_title: str) -> str:
        """Generate summary report content."""
        
        # Calculate overall statistics
        total_messages = sum(result.total_messages for result in platform_results.values())
        total_processed = sum(result.processed_messages for result in platform_results.values())
        total_tokens = sum(result.total_tokens_used for result in platform_results.values())
        total_cost = sum(result.total_cost for result in platform_results.values())
        total_time = sum(result.processing_time for result in platform_results.values())
        
        # Generate timestamp
        report_time = format_timestamp(format_type="human")
        
        # Start building report
        content = f"""# {report_title}

**生成时间**: {report_time}

## 📊 总体统计

- **总消息数**: {total_messages:,}
- **已处理消息**: {total_processed:,}
- **处理成功率**: {(total_processed/total_messages*100) if total_messages > 0 else 0:.1f}%
- **使用 Token 数**: {total_tokens:,}
- **估算成本**: ${total_cost:.4f}
- **处理时间**: {total_time:.1f} 秒

## 🌐 平台概览

"""
        
        # Add platform-specific sections
        for platform, result in platform_results.items():
            platform_name = self._get_platform_display_name(platform)
            success_rate = result.success_rate
            
            content += f"""### {platform_name}

- **消息数量**: {result.total_messages:,}
- **处理成功**: {result.processed_messages:,}
- **成功率**: {success_rate:.1f}%
- **Token 使用**: {result.total_tokens_used:,}
- **处理时间**: {result.processing_time:.1f}s

"""
            
            # Add AI analysis summary if available
            if result.summaries:
                content += "**AI 分析摘要**:\n\n"
                # Since we now have integrated results, show the full summary
                summary = result.summaries[0]  # Only one integrated summary
                preview = summary[:300] + '...' if len(summary) > 300 else summary
                content += f"{preview}\n\n"
        
        # Add error summary if any
        all_errors = []
        for result in platform_results.values():
            all_errors.extend(result.errors)
        
        if all_errors:
            content += f"""## ⚠️ 错误报告

处理过程中遇到 {len(all_errors)} 个错误:

"""
            for i, error in enumerate(all_errors[:5], 1):  # Show first 5 errors
                content += f"{i}. {error}\n"
            
            if len(all_errors) > 5:
                content += f"\n... 还有 {len(all_errors) - 5} 个错误\n"
        
        # Add footer
        content += f"""
---

*报告由 TDXAgent 自动生成 - {report_time}*
"""
        
        return content
    
    def _generate_platform_content(self, 
                                   platform: str,
                                   batch_result: BatchResult,
                                   messages: List[Dict[str, Any]],
                                   all_platform_results: Optional[Dict[str, BatchResult]] = None) -> str:
        """Generate platform-specific report content."""
        
        platform_name = self._get_platform_display_name(platform)
        report_time = format_timestamp(format_type="human")
        
        content = f"""# TDXAgent {platform_name} 数据分析报告

**生成时间**: {report_time}

## 📈 {platform_name} 处理统计

"""
        
        content += f"""- **总消息数**: {batch_result.total_messages:,}
- **成功处理**: {batch_result.processed_messages:,}
- **处理成功率**: {batch_result.success_rate:.1f}%
- **批次数量**: {batch_result.successful_batches + batch_result.failed_batches}
- **成功批次**: {batch_result.successful_batches}
- **失败批次**: {batch_result.failed_batches}
- **Token 使用**: {batch_result.total_tokens_used:,}
- **平均每批次 Token**: {batch_result.average_tokens_per_batch:.0f}
- **处理时间**: {batch_result.processing_time:.1f} 秒
- **估算成本**: ${batch_result.total_cost:.4f}

## 🤖 AI 分析结果

"""
        
        # Add integrated AI analysis result
        if batch_result.summaries:
            content += f"""### AI 综合分析

"""
            # 显示所有平台的分析摘要
            for summary in batch_result.summaries:
                content += f"{summary}\n\n"
        else:
            content += "暂无 AI 分析结果。\n\n"
        
        # 简化设计：移除复杂的引用统计和消息样本，AI分析结果中的引用链接已足够
        
        # Add platform-specific error details if any
        if batch_result.errors:
            content += f"""## ⚠️ 错误详情

处理过程中遇到以下错误：

"""
            for i, error in enumerate(batch_result.errors, 1):
                content += f"{i}. {error}\n"
        
        # Add footer
        content += f"""
---

*报告由 TDXAgent 自动生成 - {report_time}*
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
    
    async def generate_cross_platform_report(self, 
                                           analysis_result: Dict[str, Any],
                                           report_title: str = "TDXAgent 跨平台主题整合报告",
                                           start_time: Optional[datetime] = None,
                                           hours_back: Optional[int] = None) -> str:
        """
        生成跨平台主题整合报告。
        
        Args:
            analysis_result: 跨平台分析结果
            report_title: 报告标题
            start_time: 开始时间
            hours_back: 分析的小时数
            
        Returns:
            生成的报告文件路径
        """
        batch_result = analysis_result['batch_result']
        total_messages = analysis_result['total_messages_analyzed']
        platforms_included = analysis_result['platforms_included']
        platform_counts = analysis_result['platform_message_counts']
        
        # 生成报告时间
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") if start_time is None else start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # 确定时间期间描述
        if hours_back:
            if hours_back == 12:
                period_desc = "最近12小时"
            elif hours_back == 24:
                period_desc = "最近1天"
            elif hours_back == 168:  # 7*24
                period_desc = "最近1周"
            elif hours_back < 24:
                period_desc = f"最近{hours_back}小时"
            elif hours_back % 24 == 0:
                days = hours_back // 24
                period_desc = f"最近{days}天"
            else:
                period_desc = f"最近{hours_back}小时"
        else:
            period_desc = "指定时间段"
        
        # 构建跨平台报告内容
        content = f"""# {report_title}

**生成时间**: {report_time}  
**数据期间**: {period_desc}

## 📊 跨平台数据概览

| 指标 | 数值 |
|------|------|
| 📄 **总消息数** | {total_messages:,} 条 |
| 🌐 **涵盖平台数** | {len(platforms_included)} 个 |
| ✅ **处理成功** | {batch_result.processed_messages:,} 条 |
| 🎯 **处理成功率** | {batch_result.success_rate:.1f}% |
| 🤖 **Token使用量** | {batch_result.total_tokens_used:,} |
| 💰 **估算成本** | ${batch_result.total_cost:.4f} |
| ⚡ **处理耗时** | {batch_result.processing_time:.1f} 秒 |

### 📱 平台数据分布

| 平台 | 消息数量 | 占比 |
|------|----------|------|
"""
        
        # 添加平台分布统计
        for platform in platforms_included:
            count = platform_counts.get(platform, 0)
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            platform_name = self._get_platform_display_name(platform)
            content += f"| {platform_name} | {count:,} 条 | {percentage:.1f}% |\n"
        
        content += "\n"
        
        # 添加主要的AI分析结果
        if batch_result.summaries:
            content += f"""## 🤖 跨平台主题整合分析

"""
            # 显示所有平台的分析摘要，而不只是第一个
            for summary in batch_result.summaries:
                content += f"{summary}\n\n"
        
        # 添加处理统计
        content += f"""## 📈 处理统计详情

- **分析模式**: 跨平台主题整合模式
- **涵盖平台**: {', '.join([self._get_platform_display_name(p) for p in platforms_included])}
- **数据完整性**: {batch_result.success_rate:.1f}% 
- **处理批次**: {batch_result.successful_batches} 成功, {batch_result.failed_batches} 失败
- **平均处理速度**: {total_messages / batch_result.processing_time:.1f} 条/秒

"""
        
        
        # 添加报告说明
        content += f"""---

## 📋 报告说明

- **分析引擎**: 跨平台主题整合系统，按投资主题归纳多平台信息
- **数据来源**: {', '.join([self._get_platform_display_name(p) for p in platforms_included])}
- **引用追溯**: AI分析结果中的所有结论都可追溯到原始消息
- **数据安全**: 所有数据完全本地处理，未上传任何第三方服务
- **主题整合**: 自动识别跨平台相同主题信息并合并，避免重复分析

*本报告由 TDXAgent 跨平台主题整合系统自动生成 - {report_time}*
"""
        
        # 生成文件名和路径
        timestamp = datetime.now().strftime("%Y年%m月%d日_%H时%M分")
        filename = f"TDXAgent跨平台整合报告_{timestamp}_{period_desc}.md"
        
        output_path = self.output_directory / filename
        
        # 确保输出目录存在
        await ensure_directory_async(self.output_directory)
        
        # 写入文件
        await asyncio.to_thread(self._write_file, output_path, content)
        
        self.logger.info(f"Generated cross-platform report: {output_path}")
        return str(output_path)
    
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
