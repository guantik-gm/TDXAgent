"""
Intelligent batch processor for LLM requests in TDXAgent.

This module provides efficient batch processing of messages for LLM analysis,
with intelligent token management and optimization strategies.
"""

import asyncio
import math
import re
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from utils.logger import TDXLogger, log_async_function_call
from llm.base_provider import BaseLLMProvider, LLMResponse
from processors.integration_manager import IntegrationManager, AnalysisContext
from processors.prompt_manager import PromptTemplate


@dataclass
class BatchExecutionDetail:
    """详细的批次执行信息"""
    batch_number: int
    message_count: int
    tokens_used: int
    processing_time: float
    success: bool
    error_message: str = ""
    response_content: str = ""
    command_type: str = ""  # 函数名，如 "process_messages_with_template"
    llm_command: str = ""   # 具体的LLM调用命令，如 "gemini -y < xxx_prompt.txt" 或 "API调用: gpt-4"
    llm_provider: str = ""  # LLM提供商，如 "claude_cli", "openai", "gemini_cli"
    model_name: str = ""    # 模型名称，如 "gpt-4", "claude-3-sonnet"
    
    @property
    def has_meaningful_content(self) -> bool:
        """基于模板检查响应是否包含有意义的内容"""
        if not self.success or not self.response_content:
            return False
        
        # 使用基于模板的质量检测（同步版本）
        try:
            from processors.template_quality_checker import get_quality_checker
            
            # 获取质量检测器，不需要PromptManager
            quality_checker = get_quality_checker()
            
            # 直接进行同步质量检查
            result = quality_checker.is_valid_response(
                content=self.response_content,
                template_name='pure_investment_analysis',
                min_sections_required=1  # 至少包含一个章节就认为有效
            )
            
            return result.is_valid
            
        except Exception as e:
            # 如果质量检测出错，使用增强的fallback检测
            import logging
            logging.getLogger("tdxagent.batch_processor").warning(
                f"Template-based quality check failed, using enhanced fallback: {e}"
            )
            
            # 增强的fallback: 检查内容特征，避免系统消息被误判
            content = self.response_content.strip()
            
            # 基本长度检查
            if len(content) < 50:
                return False
            
            # 检查是否只是系统消息
            system_message_patterns = [
                r'^Loaded cached credentials\.',
                r'^Loading\s+.*\.\.\.',
                r'^Authenticating\s+.*\.\.\.',
                r'^Connected to\s+.*',
                r'^Error:\s+',
                r'^Warning:\s+'
            ]
            
            # 如果内容主要是系统消息，认为无效
            for pattern in system_message_patterns:
                if re.match(pattern, content, re.IGNORECASE):
                    # 检查是否还有其他有意义的内容
                    lines = content.split('\n')
                    non_empty_lines = [line.strip() for line in lines if line.strip()]
                    if len(non_empty_lines) <= 3:  # 只有很少几行，可能都是系统消息
                        return False
            
            # 检查是否包含基本的分析结构
            has_section_headers = bool(re.search(r'^##\s+', content, re.MULTILINE))
            has_analysis_content = any(keyword in content.lower() for keyword in [
                '分析', '建议', '总结', '结论', '观察', '发现', '趋势', '机会', '风险'
            ])
            
            return has_section_headers or has_analysis_content


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    platform: str
    total_messages: int
    processed_messages: int
    successful_batches: int
    failed_batches: int
    total_tokens_used: int
    total_cost: float
    processing_time: float
    summaries: List[str]
    errors: List[str]
    # 新增字段：详细的批次执行信息
    batch_details: List[BatchExecutionDetail] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.batch_details is None:
            self.batch_details = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_messages == 0:
            return 0.0
        return (self.processed_messages / self.total_messages) * 100
    
    @property
    def average_tokens_per_batch(self) -> float:
        """Calculate average tokens per successful batch."""
        if self.successful_batches == 0:
            return 0.0
        return self.total_tokens_used / self.successful_batches
    
    @property
    def failed_batch_details(self) -> List[BatchExecutionDetail]:
        """获取失败的批次详情"""
        return [detail for detail in self.batch_details if not detail.success]
    
    @property
    def empty_result_batches(self) -> List[BatchExecutionDetail]:
        """获取成功但没有有意义内容的批次"""
        return [detail for detail in self.batch_details 
                if detail.success and not detail.has_meaningful_content]
    
    @property
    def problematic_batches(self) -> List[BatchExecutionDetail]:
        """获取所有有问题的批次（失败 + 无意义内容）"""
        return self.failed_batch_details + self.empty_result_batches


@dataclass  
class IntegrationResponse:
    """整合处理响应，包含LLM响应和批次详情"""
    llm_response: 'LLMResponse'
    batch_details: List[BatchExecutionDetail]
    skipped_batches: List[int] = None  # 被跳过的批次索引
    
    @property
    def success(self) -> bool:
        return self.llm_response.success
    
    @property
    def content(self) -> str:
        return self.llm_response.content
    
    @property 
    def error_message(self) -> str:
        return self.llm_response.error_message


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_messages_per_batch: int = 200
    max_tokens_per_batch: int = 60000  # 充分利用大上下文长度
    max_concurrent_batches: int = 2  # 降低并发以避免RPM限制
    retry_failed_batches: bool = True
    max_retries: int = 2
    delay_between_batches: float = 4.5
    multi_batch_delay: float = 3.0  # 多批次处理前的缓冲时间
    token_buffer: int = 8000  # 为响应保留更多tokens
    
    # 质量重试配置
    quality_retry_enabled: bool = True  # 启用质量重试
    max_quality_retries: int = 3  # 最大质量重试次数
    quality_retry_delay: float = 10.0  # 初始重试延迟(秒)
    quality_retry_backoff_factor: float = 1.5  # 延迟递增倍数
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (
            self.max_messages_per_batch > 0 and
            self.max_tokens_per_batch > 100 and
            self.max_concurrent_batches > 0 and
            self.max_retries >= 0 and
            self.delay_between_batches >= 0 and
            self.token_buffer >= 0 and
            self.max_quality_retries >= 0 and
            self.quality_retry_delay >= 0 and
            self.quality_retry_backoff_factor > 0
        )


class BatchProcessor:
    """
    Intelligent batch processor for LLM requests.
    
    Features:
    - Smart batching based on token limits
    - Concurrent processing with rate limiting
    - Automatic retry for failed batches
    - Token usage optimization
    - Progress tracking and reporting
    """
    
    def __init__(self, llm_provider: BaseLLMProvider, config: Optional[BatchConfig] = None, llm_config=None):
        """
        Initialize batch processor.
        
        Args:
            llm_provider: LLM provider instance
            config: Batch processing configuration
            llm_config: LLM configuration from config manager
        """
        self.llm_provider = llm_provider
        
        # Create batch config from LLM config if provided
        if config is None and llm_config is not None:
            config = BatchConfig(
                max_messages_per_batch=llm_config.batch_size,
                max_tokens_per_batch=999999,  # 不限制Token，只按消息数量分批
                max_concurrent_batches=2,
                retry_failed_batches=True,
                max_retries=llm_config.max_retries,
                delay_between_batches=llm_config.delay_between_batches,
                multi_batch_delay=llm_config.multi_batch_delay,  # 多批次处理前的缓冲时间
                token_buffer=0  # 不需要Token缓冲
            )
        
        self.config = config or BatchConfig()
        self.logger = TDXLogger.get_logger("tdxagent.processors.batch")
        
        if not self.config.validate():
            raise ValueError("Invalid batch configuration")
        
        # Initialize integration manager
        self.integration_manager = IntegrationManager(llm_provider=llm_provider)
        
        # Processing state
        self._processing_stats = {
            'total_batches_processed': 0,
            'total_messages_processed': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0,
            'total_processing_time': 0.0
        }
        
        # Integration manager for progressive analysis
        self.integration_manager = IntegrationManager()
        
        
        self.logger.info(f"Initialized batch processor with progressive integration support")
    
    async def process_messages(self, messages: List[Dict[str, Any]], 
                              prompt_template: str,
                              platform: str = "",
                              progress_callback: Optional[Callable[[int, int], None]] = None) -> BatchResult:
        """
        Process messages in intelligent batches.
        
        Args:
            messages: List of message dictionaries
            prompt_template: Template with {data} placeholder
            platform: Platform name for context
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with processing results
        """
        start_time = datetime.now()
        
        if not messages:
            return BatchResult(
                platform=platform,
                total_messages=0,
                processed_messages=0,
                successful_batches=0,
                failed_batches=0,
                total_tokens_used=0,
                total_cost=0.0,
                processing_time=0.0,
                summaries=[],
                errors=[]
            )
        
        self.logger.info(f"Starting batch processing of {len(messages)} messages")
        
        # Create intelligent batches
        batches = self._create_intelligent_batches(messages, prompt_template)
        self.logger.info(f"Created {len(batches)} batches for processing")
        
        # Process batches with progressive integration
        final_result = await self._process_batches_with_integration(
            batches, prompt_template, platform, progress_callback
        )
        
        # Compile final results
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Count successful and failed batches from integration result
        successful_batches = len(batches) if final_result.success else 0
        failed_batches = len(batches) - successful_batches
        
        batch_result = BatchResult(
            platform=platform,
            total_messages=len(messages),
            processed_messages=len(messages) if final_result.success else 0,
            successful_batches=successful_batches,
            failed_batches=failed_batches,
            total_tokens_used=final_result.token_count if final_result.success else 0,
            total_cost=self.llm_provider.calculate_cost(final_result.usage) if (
                final_result.success and hasattr(self.llm_provider, 'calculate_cost')
            ) else 0.0,
            processing_time=processing_time,
            summaries=[final_result.content] if final_result.success else [],
            errors=[final_result.error_message] if not final_result.success else []
        )
        
        # Update global stats
        self._update_stats(batch_result)
        
        self.logger.info(
            f"Batch processing complete: {batch_result.processed_messages}/{batch_result.total_messages} "
            f"messages processed in {processing_time:.2f}s "
            f"({batch_result.success_rate:.1f}% success rate)"
        )
        
        return batch_result
    
    def _create_intelligent_batches(self, messages: List[Dict[str, Any]], 
                                   prompt_template: str) -> List[List[Dict[str, Any]]]:
        """
        Create simple batches based on message count only.
        
        Args:
            messages: List of messages to batch
            prompt_template: Prompt template (unused, kept for compatibility)
            
        Returns:
            List of message batches
        """
        batches = []
        current_batch = []
        
        for message in messages:
            # Check if batch is full
            if len(current_batch) >= self.config.max_messages_per_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = []
            
            # Add message to current batch
            current_batch.append(message)
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _format_message_for_estimation(self, message: Dict[str, Any]) -> str:
        """格式化单个消息以进行token估算（适配优化数据结构）。"""
        author = message.get('author', {}).get('name', 'Unknown')
        content = message.get('content', {}).get('text', '')
        timestamp = message.get('metadata', {}).get('posted_at', '')
        platform = message.get('platform', '')
        
        return f"[{timestamp}] {author} ({platform}): {content}"
    
    async def _process_batches_with_integration(self, batches: List[List[Dict[str, Any]]], 
                                              prompt_template: str,
                                              platform: str = "",
                                              progress_callback: Optional[Callable[[int, int], None]] = None) -> LLMResponse:
        """
        Process batches with progressive integration.
        
        Args:
            batches: List of message batches
            prompt_template: Base prompt template
            platform: Platform name for template selection
            progress_callback: Progress callback function
            
        Returns:
            Single integrated LLM response
        """
        if not batches:
            return LLMResponse(
                content="",
                usage={},
                model=self.llm_provider.default_model,
                provider=self.llm_provider.provider_name,
                timestamp=datetime.now(),
                success=False,
                error_message="No batches to process"
            )
        
        self.logger.info(f"Starting progressive integration of {len(batches)} batches")
        
        # 多批次处理前的缓冲时间 - 提升AI响应成功率
        if len(batches) > 1:
            buffer_delay = self.config.multi_batch_delay
            self.logger.info(f"检测到多批次处理 ({len(batches)}个批次)，等待 {buffer_delay} 秒缓冲时间...")
            await asyncio.sleep(buffer_delay)
        
        current_analysis = None
        current_context = None
        total_tokens_used = 0
        accumulated_usage = {}
        
        # Process each batch progressively
        for batch_index, batch in enumerate(batches):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(batch_index + 1, len(batches))
                
                # Determine if this is an integration batch
                is_integration = self.integration_manager.should_use_integration_mode(
                    batch_index + 1, len(batches)
                )
                
                if is_integration and current_analysis:
                    # Use integration template and context
                    result = await self._process_integration_batch(
                        batch, prompt_template, current_analysis, platform
                    )
                else:
                    # Use regular template for first batch
                    result = await self._process_batch_with_quality_retry(batch, prompt_template, platform, batch_index)
                
                if not result.success:
                    self.logger.error(f"Batch {batch_index + 1} failed: {result.error_message}")
                    return result  # Return failed result immediately
                
                # 创建BatchExecutionDetail进行质量检测
                batch_detail = BatchExecutionDetail(
                    batch_number=batch_index + 1,
                    message_count=len(batch),
                    tokens_used=result.usage.get('total_tokens', 0),
                    processing_time=0,  # 时间在_process_single_batch中已计算
                    success=True,
                    response_content=result.content,
                    command_type="process_batches_with_integration",
                    llm_command=getattr(result, 'call_command', 'command not available')
                )
                
                # 检查响应内容质量
                if not batch_detail.has_meaningful_content:
                    self.logger.warning(f"平台 {platform} 批次 {batch_index + 1} 返回内容质量不佳或无有效结果")
                    self.logger.warning(f"LLM命令: {batch_detail.llm_command}")
                    
                    # 记录完整响应内容用于调试
                    if len(result.content) > 1000:
                        self.logger.warning(f"完整响应内容 (前1000字符): {result.content[:1000]}...")
                        self.logger.warning(f"完整响应内容 (后500字符): ...{result.content[-500:]}")
                    else:
                        self.logger.warning(f"完整响应内容: {result.content}")
                    
                    # 如果响应对象有错误信息或额外信息，也记录下来
                    if hasattr(result, 'error_message') and result.error_message:
                        self.logger.warning(f"响应错误信息: {result.error_message}")
                    
                    # 记录响应元数据
                    self.logger.warning(f"响应成功状态: {result.success}")
                    self.logger.warning(f"响应Token使用: {result.usage.get('total_tokens', 'unknown')}")
                    self.logger.warning(f"响应提供商: {result.provider}")
                    self.logger.warning(f"响应模型: {result.model}")
                    
                    # 对于无效响应，我们有几个选择：
                    # 1. 跳过这个批次，不更新current_analysis
                    # 2. 返回错误
                    # 3. 使用空的分析结果
                    
                    # 对于第一个批次，如果无效则返回错误
                    if batch_index == 0:
                        self.logger.error(f"第一个批次返回无效内容，无法继续处理")
                        return LLMResponse(
                            content="",
                            usage=accumulated_usage,
                            model=self.llm_provider.default_model,
                            provider=self.llm_provider.provider_name,
                            timestamp=datetime.now(),
                            success=False,
                            error_message=f"第一个批次返回无效内容: {content_preview}"
                        )
                    
                    # 对于后续批次，跳过并保持current_analysis不变
                    self.logger.info(f"跳过批次 {batch_index + 1}，保持前批次分析结果")
                    continue
                
                # Update current analysis and context
                current_analysis = result.content
                if batch_index < len(batches) - 1:  # Not the last batch
                    # Prepare context for next batch
                    context_token_limit = self.integration_manager.calculate_context_token_limit(
                        self.config.max_tokens_per_batch, self.config.token_buffer
                    )
                    current_context = self.integration_manager.compress_analysis_context(
                        current_analysis, context_token_limit
                    )
                
                # Accumulate usage statistics
                total_tokens_used += result.token_count
                if result.usage:
                    for key, value in result.usage.items():
                        accumulated_usage[key] = accumulated_usage.get(key, 0) + value
                
                # Add delay between batches
                if batch_index < len(batches) - 1:
                    await asyncio.sleep(self.config.delay_between_batches)
                
                self.logger.info(f"Completed batch {batch_index + 1}/{len(batches)} "
                               f"({'integration' if is_integration else 'initial'})")
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_index + 1}: {e}")
                return LLMResponse(
                    content="",
                    usage=accumulated_usage,
                    model=self.llm_provider.default_model,
                    provider=self.llm_provider.provider_name,
                    timestamp=datetime.now(),
                    success=False,
                    error_message=f"Integration processing failed: {str(e)}"
                )
        
        # Return final integrated result
        final_result = LLMResponse(
            content=current_analysis or "",
            usage=accumulated_usage,
            model=self.llm_provider.default_model,
            provider=self.llm_provider.provider_name,
            timestamp=datetime.now(),
            success=True,
            error_message=None
        )
        
        # Validate final result
        is_valid, validation_error = self.integration_manager.validate_integration_result(current_analysis)
        if not is_valid:
            self.logger.warning(f"Integration result validation failed: {validation_error}")
            # Still return the result but log the warning
        
        self.logger.info(f"Progressive integration completed successfully. "
                        f"Total tokens: {total_tokens_used}")
        
        return final_result

    async def _process_batches_concurrent(self, batches: List[List[Dict[str, Any]]], 
                                        prompt_template: str,
                                        progress_callback: Optional[Callable[[int, int], None]] = None) -> List[LLMResponse]:
        """
        Process batches with controlled concurrency.
        
        Args:
            batches: List of message batches
            prompt_template: Prompt template
            progress_callback: Progress callback function
            
        Returns:
            List of LLM responses
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        results = []
        completed = 0
        
        async def process_single_batch(batch_index: int, batch: List[Dict[str, Any]]) -> LLMResponse:
            async with semaphore:
                nonlocal completed
                
                try:
                    # Add delay between batches to respect rate limits
                    if batch_index > 0:
                        await asyncio.sleep(self.config.delay_between_batches)
                    
                    # Process the batch
                    result = await self._process_single_batch(batch, prompt_template)
                    
                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(batches))
                    
                    self.logger.debug(f"Completed batch {batch_index + 1}/{len(batches)}")
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Batch {batch_index + 1} failed: {e}")
                    return LLMResponse(
                        content="",
                        usage={},
                        model=self.llm_provider.default_model,
                        provider=self.llm_provider.provider_name,
                        timestamp=datetime.now(),
                        success=False,
                        error_message=str(e)
                    )
        
        # Create tasks for all batches
        tasks = [
            process_single_batch(i, batch) 
            for i, batch in enumerate(batches)
        ]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch {i + 1} raised exception: {result}")
                processed_results.append(LLMResponse(
                    content="",
                    usage={},
                    model=self.llm_provider.default_model,
                    provider=self.llm_provider.provider_name,
                    timestamp=datetime.now(),
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_integration_batch(self, batch: List[Dict[str, Any]], 
                                       prompt_template: str,
                                       previous_analysis: str,
                                       platform: str = "") -> LLMResponse:
        """
        Process a batch with integration context from previous analysis.
        
        Args:
            batch: List of messages in the batch
            prompt_template: Base prompt template
            previous_analysis: Raw analysis result from previous batch
            platform: Platform name for template selection
            
        Returns:
            LLM response with integrated analysis
        """
        try:
            # Create integration context for prompt using raw analysis text
            integration_context = self.integration_manager.create_integration_prompt_context(
                previous_analysis, platform
            )
            
            # Import here to avoid circular dependency
            from utils.link_generator import LinkGenerator
            
            # Format messages using unified method - platform agnostic
            link_generator = LinkGenerator()
            formatted_messages = link_generator.format_messages_unified(batch)
            
            # Create the integration prompt using the original template
            prompt = prompt_template.format(
                data=formatted_messages,
                integration_context=integration_context
            )
            
            # 预估token数量用于日志
            estimated_tokens = self.llm_provider.estimate_tokens(prompt)
            
            self.logger.info(f"处理整合批次 {len(batch)} 条消息 (平台: {platform}, 预估 {estimated_tokens} tokens)")
            
            # Make the request with retry logic
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await self.llm_provider.generate_response(prompt, platform=platform)
                    
                    if response.success:
                        self.logger.debug(f"Integration batch completed with {response.token_count} tokens")
                        return response
                    elif attempt < self.config.max_retries:
                        self.logger.warning(f"Integration batch attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(self.config.delay_between_batches * (attempt + 1))
                    
                except Exception as e:
                    if attempt < self.config.max_retries:
                        self.logger.warning(f"Integration batch attempt {attempt + 1} error: {e}, retrying...")
                        await asyncio.sleep(self.config.delay_between_batches * (attempt + 1))
                    else:
                        raise
            
            # If we get here, all retries failed
            return LLMResponse(
                content="",
                usage={},
                model=self.llm_provider.default_model,
                provider=self.llm_provider.provider_name,
                timestamp=datetime.now(),
                success=False,
                error_message="All integration retry attempts failed"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process integration batch: {e}")
            return LLMResponse(
                content="",
                usage={},
                model=self.llm_provider.default_model,
                provider=self.llm_provider.provider_name,
                timestamp=datetime.now(),
                success=False,
                error_message=f"Integration batch processing error: {str(e)}"
            )
    
    async def _process_single_batch(self, batch: List[Dict[str, Any]], 
                                   prompt_template: str, 
                                   platform: str = "") -> LLMResponse:
        """
        Process a single batch of messages.
        
        Args:
            batch: List of messages in the batch
            prompt_template: Prompt template
            
        Returns:
            LLM response
        """
        # Import here to avoid circular dependency
        from utils.link_generator import LinkGenerator
        
        # Format messages using unified method - platform agnostic
        link_generator = LinkGenerator()
        formatted_messages = link_generator.format_messages_unified(batch)
        
        # Create the prompt with all required variables
        prompt = prompt_template.format(
            data=formatted_messages,
            integration_context=""  # Empty for regular analysis
        )
        
        # 预估token数量用于日志
        estimated_tokens = self.llm_provider.estimate_tokens(prompt)
        
        self.logger.info(f"处理单批次 {len(batch)} 条消息 (预估 {estimated_tokens} tokens)")
        
        # Make the request with retry logic
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.llm_provider.generate_response(prompt, platform=platform)
                
                if response.success:
                    return response
                elif attempt < self.config.max_retries:
                    self.logger.warning(f"Batch attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(self.config.delay_between_batches * (attempt + 1))
                
            except Exception as e:
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Batch attempt {attempt + 1} error: {e}, retrying...")
                    await asyncio.sleep(self.config.delay_between_batches * (attempt + 1))
                else:
                    raise
        
        # If we get here, all retries failed
        return LLMResponse(
            content="",
            usage={},
            model=self.llm_provider.default_model,
            provider=self.llm_provider.provider_name,
            timestamp=datetime.now(),
            success=False,
            error_message="All retry attempts failed"
        )
    
    async def _process_batch_with_quality_retry(self, batch: List[Dict[str, Any]], 
                                               prompt_template: str, 
                                               platform: str = "",
                                               batch_index: int = 0) -> LLMResponse:
        """
        Process a batch with intelligent quality retry mechanism.
        
        Args:
            batch: List of messages in the batch
            prompt_template: Prompt template
            platform: Platform name for logging
            batch_index: Index of the batch for logging
            
        Returns:
            LLM response with quality validation
        """
        if not self.config.quality_retry_enabled:
            # 如果质量重试未启用，使用原有逻辑
            self.logger.info(f"批次 {batch_index + 1} 质量重试被禁用，使用直接处理")
            return await self._process_single_batch(batch, prompt_template, platform)
        
        self.logger.info(f"批次 {batch_index + 1} 开始质量重试处理，最大重试次数: {self.config.max_quality_retries}")
        
        retry_delay = self.config.quality_retry_delay
        
        for retry_attempt in range(self.config.max_quality_retries + 1):
            # 执行批次处理
            response = await self._process_single_batch(batch, prompt_template, platform)
            
            # 如果LLM调用本身失败，不进行质量重试，直接返回失败
            if not response.success:
                self.logger.warning(f"批次 {batch_index + 1} LLM调用失败，不进行质量重试: {response.error_message}")
                return response
            
            # 创建BatchExecutionDetail来检查内容质量
            batch_detail = BatchExecutionDetail(
                batch_number=batch_index + 1,
                message_count=len(batch),
                success=response.success,
                tokens_used=response.usage.get('total_tokens', 0),
                processing_time=0.0,  # 这里不计算时间，主要用于质量检测
                response_content=response.content,
                llm_command=getattr(response, 'call_command', 'command not available'),
                command_type="process_batch_with_quality_retry"
            )
            
            # 检查响应质量
            if batch_detail.has_meaningful_content:
                if retry_attempt > 0:
                    self.logger.info(f"批次 {batch_index + 1} 质量重试第 {retry_attempt} 次成功")
                return response
            
            # 内容质量不佳，考虑重试
            if retry_attempt < self.config.max_quality_retries:
                self.logger.warning(f"批次 {batch_index + 1} 质量检测失败，将在 {retry_delay:.1f} 秒后进行第 {retry_attempt + 1} 次重试")
                
                # 详细记录问题响应内容用于调试
                if len(response.content) > 1000:
                    self.logger.warning(f"问题响应内容 (前1000字符): {response.content[:1000]}...")
                    self.logger.warning(f"问题响应内容 (后500字符): ...{response.content[-500:]}")
                else:
                    self.logger.warning(f"问题响应完整内容: {response.content}")
                
                # 记录响应元数据
                if hasattr(response, 'error_message') and response.error_message:
                    self.logger.warning(f"响应错误信息: {response.error_message}")
                self.logger.warning(f"LLM命令: {getattr(response, 'call_command', 'unknown')}")
                
                # 等待重试延迟
                await asyncio.sleep(retry_delay)
                
                # 递增延迟时间
                retry_delay *= self.config.quality_retry_backoff_factor
            else:
                # 所有重试都失败了
                self.logger.error(f"批次 {batch_index + 1} 经过 {self.config.max_quality_retries} 次质量重试仍然失败")
                
                # 记录完整的最终响应内容
                if len(response.content) > 1000:
                    self.logger.error(f"最终响应内容 (前1000字符): {response.content[:1000]}...")
                    self.logger.error(f"最终响应内容 (后500字符): ...{response.content[-500:]}")
                else:
                    self.logger.error(f"最终响应完整内容: {response.content}")
                
                # 记录响应元数据
                if hasattr(response, 'error_message') and response.error_message:
                    self.logger.error(f"最终响应错误信息: {response.error_message}")
                self.logger.error(f"最终LLM命令: {getattr(response, 'call_command', 'unknown')}")
                
                # 返回一个标记为失败的响应
                return LLMResponse(
                    content="",
                    usage=response.usage,
                    model=response.model,
                    provider=response.provider,
                    timestamp=response.timestamp,
                    success=False,
                    error_message=f"质量重试失败: 响应内容质量不佳，经过 {self.config.max_quality_retries} 次重试仍无法获得有效结果"
                )
        
        # 理论上不应该到达这里
        return response
    
    def _update_stats(self, batch_result: BatchResult) -> None:
        """Update global processing statistics."""
        self._processing_stats['total_batches_processed'] += (
            batch_result.successful_batches + batch_result.failed_batches
        )
        self._processing_stats['total_messages_processed'] += batch_result.processed_messages
        self._processing_stats['total_tokens_used'] += batch_result.total_tokens_used
        self._processing_stats['total_cost'] += batch_result.total_cost
        self._processing_stats['total_processing_time'] += batch_result.processing_time
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        return self._processing_stats.copy()
    
    def get_collected_conversations(self) -> List[Dict[str, Any]]:
        """Get collected LLM conversations for report appendix.
        
        This method retrieves complete LLM conversations that were automatically
        collected at the provider interface level. Each conversation includes
        request prompt, response content, and full metadata.
        
        Returns:
            List of conversation dictionaries with complete request-response data
        """
        conversations = self.llm_provider.get_collected_conversations()
        return [conv.to_dict() for conv in conversations]
    
    def clear_collected_conversations(self) -> None:
        """Clear collected conversations after report generation.
        
        Clears the conversation collection in the LLM provider to free memory
        and prepare for the next analysis cycle. Should be called after
        generating reports to avoid accumulating conversations across sessions.
        """
        self.llm_provider.clear_collected_conversations()
    
    def enable_conversation_collection(self) -> None:
        """Enable LLM conversation collection for debugging.
        
        Activates the interface-level conversation recording in the LLM provider.
        This captures every request-response pair with full metadata including
        tokens, cost, timing, and success status for debugging purposes.
        """
        self.llm_provider.enable_conversation_collection()
    
    def disable_conversation_collection(self) -> None:
        """Disable LLM conversation collection."""
        self.llm_provider.disable_conversation_collection()
    
    def estimate_processing_cost(self, messages: List[Dict[str, Any]], 
                                prompt_template: str) -> Dict[str, Any]:
        """
        Estimate the cost and time for processing messages.
        
        Args:
            messages: List of messages to process
            prompt_template: Prompt template
            
        Returns:
            Dictionary with cost and time estimates
        """
        if not messages:
            return {
                'estimated_batches': 0,
                'estimated_tokens': 0,
                'estimated_cost': 0.0,
                'estimated_time_seconds': 0.0
            }
        
        # Create batches to get accurate estimates
        batches = self._create_intelligent_batches(messages, prompt_template)
        
        # Estimate tokens for each batch
        total_tokens = 0
        for batch in batches:
            formatted_messages = self.llm_provider.format_messages_for_prompt(batch)
            prompt = prompt_template.format(
                data=formatted_messages,
                integration_context=""  # Empty for estimation
            )
            batch_tokens = self.llm_provider.estimate_tokens(prompt)
            batch_tokens += self.config.token_buffer  # Add response tokens
            total_tokens += batch_tokens
        
        # Estimate cost if provider supports it
        estimated_cost = 0.0
        if hasattr(self.llm_provider, 'calculate_cost'):
            # Rough estimate assuming 80% input, 20% output tokens
            input_tokens = int(total_tokens * 0.8)
            output_tokens = int(total_tokens * 0.2)
            usage = {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': total_tokens
            }
            estimated_cost = self.llm_provider.calculate_cost(usage)
        
        # Estimate processing time
        estimated_time = len(batches) * self.config.delay_between_batches
        estimated_time += len(batches) * 5  # Rough estimate of 5 seconds per batch
        
        return {
            'estimated_batches': len(batches),
            'estimated_tokens': total_tokens,
            'estimated_cost': estimated_cost,
            'estimated_time_seconds': estimated_time
        }
    
    async def process_messages_with_template(self, 
                                           messages: List[Dict[str, Any]], 
                                           prompt_template: PromptTemplate,
                                           platform: str = "",
                                           formatted_messages: str = "",
                                           progress_callback: Optional[Callable[[int, int], None]] = None) -> BatchResult:
        """
        Process messages using PromptTemplate and formatted message citations.
        
        Args:
            messages: List of message dictionaries
            prompt_template: PromptTemplate instance from PromptManager
            platform: Platform name for integration templates
            formatted_messages: Pre-formatted messages with citation references
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with analysis results
        """
        if not messages:
            self.logger.warning("No messages to process")
            return BatchResult(
                platform=platform,
                total_messages=0,
                processed_messages=0,
                successful_batches=0,
                failed_batches=0,
                total_tokens_used=0,
                total_cost=0.0,
                processing_time=0.0,
                summaries=[],
                errors=["No messages provided"]
            )
        
        start_time = time.time()
        
        # Use the template string from PromptTemplate
        template_string = prompt_template.template
        
        # If formatted_messages is provided, use it directly; otherwise format normally
        if formatted_messages:
            # Use the formatted messages with citations
            final_prompt = template_string.format(
                data=formatted_messages,
                integration_context=""  # Empty for formatted messages
            )
            
            # Process as single batch since messages are already formatted with citations
            batch_detail = BatchExecutionDetail(
                batch_number=1,
                message_count=len(messages),
                tokens_used=0,
                processing_time=0,
                success=False,
                command_type="process_messages_with_template"
            )
            
            try:
                response = await self.llm_provider.generate_response(final_prompt, platform=platform)
                
                processing_time = time.time() - start_time
                batch_detail.processing_time = processing_time
                batch_detail.tokens_used = response.usage.get('total_tokens', 0)
                batch_detail.response_content = response.content
                batch_detail.llm_command = getattr(response, 'call_command', 'command not available')
                
                if response.success:
                    batch_detail.success = True
                    
                    # 检查响应内容质量并记录
                    if not batch_detail.has_meaningful_content:
                        self.logger.warning(f"平台 {platform} 批次 1 返回内容质量不佳或无有效结果")
                        batch_detail.error_message = "AI返回内容无有效结果或质量不佳"
                    
                    result = BatchResult(
                        platform=platform,
                        total_messages=len(messages),
                        processed_messages=len(messages),
                        successful_batches=1,
                        failed_batches=0,
                        total_tokens_used=response.usage.get('total_tokens', 0),
                        total_cost=self.llm_provider.calculate_cost(response.usage),
                        processing_time=processing_time,
                        summaries=[response.content],
                        errors=[],
                        batch_details=[batch_detail]
                    )
                    
                    # 记录详细的执行结果
                    self._log_batch_execution_results(result)
                    return result
                else:
                    batch_detail.success = False
                    batch_detail.error_message = response.error_message
                    batch_detail.llm_command = getattr(response, 'call_command', 'command not available')
                    
                    self.logger.error(f"平台 {platform} 批次 1 处理失败: {response.error_message}")
                    
                    result = BatchResult(
                        platform=platform,
                        total_messages=len(messages),
                        processed_messages=0,
                        successful_batches=0,
                        failed_batches=1,
                        total_tokens_used=0,
                        total_cost=0.0,
                        processing_time=processing_time,
                        summaries=[],
                        errors=[f"LLM processing failed: {response.error_message}"],
                        batch_details=[batch_detail]
                    )
                    
                    self._log_batch_execution_results(result)
                    return result
                    
            except Exception as e:
                processing_time = time.time() - start_time
                batch_detail.processing_time = processing_time
                batch_detail.success = False
                batch_detail.error_message = str(e)
                batch_detail.llm_command = "exception occurred before command execution"
                
                self.logger.error(f"平台 {platform} 批次 1 处理异常: {e}")
                
                result = BatchResult(
                    platform=platform,
                    total_messages=len(messages),
                    processed_messages=0,
                    successful_batches=0,
                    failed_batches=1,
                    total_tokens_used=0,
                    total_cost=0.0,
                    processing_time=processing_time,
                    summaries=[],
                    errors=[f"Processing error: {str(e)}"],
                    batch_details=[batch_detail]
                )
                
                self._log_batch_execution_results(result)
                return result
        else:
            # No formatted messages provided - need to do batch processing with platform formatting
            # Import here to avoid circular dependency
            from utils.link_generator import LinkGenerator
            
            # Create LinkGenerator for platform-specific formatting
            link_generator = LinkGenerator()
            
            # Create intelligent batches
            batches = self._create_intelligent_batches(messages, template_string)
            self.logger.info(f"Created {len(batches)} batches for {platform} processing")
            
            # If only one batch, process directly without integration
            if len(batches) == 1:
                # Single batch - process without integration
                batch_messages = batches[0]
                try:
                    # Format this batch of messages using unified method - platform agnostic
                    formatted_batch = link_generator.format_messages_unified(batch_messages)
                    
                    # Create prompt for this batch
                    batch_prompt = template_string.format(
                        data=formatted_batch,
                        integration_context=""
                    )
                    
                    # 添加单批次处理的详细日志
                    estimated_tokens = self.llm_provider.estimate_tokens(batch_prompt)
                    self.logger.info(f"处理单批次 {len(batch_messages)} 条消息 (预估 {estimated_tokens} tokens)")
                    
                    # 使用质量重试机制处理单批次
                    if self.config.quality_retry_enabled:
                        retry_delay = self.config.quality_retry_delay
                        
                        for retry_attempt in range(self.config.max_quality_retries + 1):
                            # Process this batch
                            response = await self.llm_provider.generate_response(batch_prompt, platform=platform)
                            
                            # 如果LLM调用本身失败，不进行质量重试
                            if not response.success:
                                self.logger.warning(f"单批次 LLM调用失败，不进行质量重试: {response.error_message}")
                                break
                            
                            # 创建BatchExecutionDetail来检测质量
                            batch_detail = BatchExecutionDetail(
                                batch_number=1,
                                message_count=len(batch_messages),
                                success=response.success,
                                tokens_used=response.usage.get('total_tokens', 0),
                                processing_time=0.0,
                                response_content=response.content,
                                llm_command=getattr(response, 'call_command', 'command not available'),
                                command_type="process_messages_with_template"
                            )
                            
                            # 检查响应质量
                            if batch_detail.has_meaningful_content:
                                if retry_attempt > 0:
                                    self.logger.info(f"单批次质量重试第 {retry_attempt} 次成功")
                                break
                            
                            # 内容质量不佳，考虑重试
                            if retry_attempt < self.config.max_quality_retries:
                                self.logger.warning(f"单批次质量检测失败，将在 {retry_delay:.1f} 秒后进行第 {retry_attempt + 1} 次重试")
                                self.logger.warning(f"问题响应内容预览: {response.content[:200]}...")
                                
                                # 等待重试延迟
                                await asyncio.sleep(retry_delay)
                                
                                # 递增延迟时间
                                retry_delay *= self.config.quality_retry_backoff_factor
                            else:
                                # 所有重试都失败了
                                self.logger.error(f"单批次经过 {self.config.max_quality_retries} 次质量重试仍然失败")
                                self.logger.error(f"最终响应内容: {response.content[:500]}...")
                                
                                # 标记为失败
                                response = LLMResponse(
                                    content="",
                                    usage=response.usage,
                                    model=response.model,
                                    provider=response.provider,
                                    timestamp=response.timestamp,
                                    success=False,
                                    error_message=f"质量重试失败: 响应内容质量不佳，经过 {self.config.max_quality_retries} 次重试仍无法获得有效结果"
                                )
                                break
                    else:
                        # 如果质量重试未启用，使用原有逻辑
                        response = await self.llm_provider.generate_response(batch_prompt, platform=platform)
                    
                    processing_time = time.time() - start_time
                    
                    if response.success:
                        return BatchResult(
                            platform=platform,
                            total_messages=len(messages),
                            processed_messages=len(batch_messages),
                            successful_batches=1,
                            failed_batches=0,
                            total_tokens_used=getattr(response, 'tokens_used', response.token_count if hasattr(response, 'token_count') else 0),
                            total_cost=getattr(response, 'cost', 0.0),
                            processing_time=processing_time,
                            summaries=[response.content],
                            errors=[]
                        )
                    else:
                        return BatchResult(
                            platform=platform,
                            total_messages=len(messages),
                            processed_messages=0,
                            successful_batches=0,
                            failed_batches=1,
                            total_tokens_used=0,
                            total_cost=0.0,
                            processing_time=processing_time,
                            summaries=[],
                            errors=[f"LLM processing failed: {response.error_message}"]
                        )
                        
                except Exception as e:
                    processing_time = time.time() - start_time
                    self.logger.error(f"Error processing single batch: {e}")
                    return BatchResult(
                        platform=platform,
                        total_messages=len(messages),
                        processed_messages=0,
                        successful_batches=0,
                        failed_batches=1,
                        total_tokens_used=0,
                        total_cost=0.0,
                        processing_time=processing_time,
                        summaries=[],
                        errors=[f"Processing error: {str(e)}"]
                    )
            else:
                # Multiple batches - use integration analysis
                integration_response = await self._process_batches_with_integration(
                    batches, template_string, platform, progress_callback
                )
                
                # Convert LLMResponse to BatchResult
                processing_time = time.time() - start_time
                
                if integration_response.success:
                    # 创建成功的批次详情 - 这样报告生成器能检测到被跳过的无效批次
                    # 注意：这是简化的实现，实际的批次处理详情在_process_batches_with_integration中
                    successful_batch_details = []
                    for i, batch in enumerate(batches):
                        batch_detail = BatchExecutionDetail(
                            batch_number=i + 1,
                            message_count=len(batch),
                            tokens_used=getattr(integration_response, 'tokens_used', 0) // len(batches),  # 平均分配
                            processing_time=processing_time / len(batches),  # 平均分配时间
                            success=True,  # 整体成功
                            response_content=integration_response.content if i == len(batches) - 1 else "",  # 只有最后一个批次有完整内容
                            command_type="process_batches_with_integration",
                            llm_command=getattr(integration_response, 'call_command', 'integration processing')
                        )
                        successful_batch_details.append(batch_detail)
                    
                    return BatchResult(
                        platform=platform,
                        total_messages=len(messages),
                        processed_messages=len(messages),
                        successful_batches=len(batches),
                        failed_batches=0,
                        total_tokens_used=getattr(integration_response, 'tokens_used', integration_response.token_count if hasattr(integration_response, 'token_count') else 0),
                        total_cost=getattr(integration_response, 'cost', 0.0),
                        processing_time=processing_time,
                        summaries=[integration_response.content],
                        errors=[],
                        batch_details=successful_batch_details  # 添加批次详情
                    )
                else:
                    # 创建失败的批次详情 - 确保报告中能显示失败信息
                    failed_batch_details = []
                    for i, batch in enumerate(batches):
                        batch_detail = BatchExecutionDetail(
                            batch_number=i + 1,
                            message_count=len(batch),
                            tokens_used=0,
                            processing_time=processing_time / len(batches),  # 平均分配时间
                            success=False,
                            response_content="",
                            command_type="process_batches_with_integration",
                            llm_command=getattr(integration_response, 'call_command', 'integration failed'),
                            error_message=integration_response.error_message
                        )
                        failed_batch_details.append(batch_detail)
                    
                    return BatchResult(
                        platform=platform,
                        total_messages=len(messages),
                        processed_messages=0,
                        successful_batches=0,
                        failed_batches=len(batches),
                        total_tokens_used=0,
                        total_cost=0.0,
                        processing_time=processing_time,
                        summaries=[],
                        errors=[f"Integration processing failed: {integration_response.error_message}"],
                        batch_details=failed_batch_details  # 添加详细的批次信息
                    )
    
    async def process_cross_platform_analysis(self, 
                                            all_platform_data: Dict[str, List[Dict[str, Any]]], 
                                            prompt_template: PromptTemplate,
                                            formatted_all_data: str,
                                            progress_callback: Optional[Callable[[int, int], None]] = None) -> BatchResult:
        """
        执行跨平台主题整合分析。
        
        Args:
            all_platform_data: 所有平台的消息数据
            prompt_template: 提示词模板
            formatted_all_data: 格式化后的跨平台数据
            progress_callback: 进度回调函数
            
        Returns:
            BatchResult 跨平台分析结果
        """
        # 计算总消息数
        total_messages = sum(len(messages) for messages in all_platform_data.values())
        
        if total_messages == 0:
            self.logger.warning("No messages to process for cross-platform analysis")
            return BatchResult(
                platform="cross_platform",
                total_messages=0,
                processed_messages=0,
                successful_batches=0,
                failed_batches=0,
                total_tokens_used=0,
                total_cost=0.0,
                processing_time=0.0,
                summaries=[],
                errors=["No messages provided for cross-platform analysis"]
            )
        
        start_time = time.time()
        
        # 使用提示词模板
        template_string = prompt_template.template
        
        self.logger.info(f"开始跨平台主题整合分析: {total_messages} 条消息来自 {len(all_platform_data)} 个平台")
        
        # 准备模板变量
        template_vars = {
            'data': formatted_all_data,
            'integration_context': ''  # 暂时为空，未来可支持批次间整合
        }
        
        try:
            # 使用统一的 LLM 处理方法
            filled_template = template_string.format(**template_vars)
            
            # 调用 LLM 进行跨平台分析
            response = await self.llm_provider.generate_response(filled_template, platform="cross_platform")
            
            if response.success:
                processing_time = time.time() - start_time
                
                result = BatchResult(
                    platform="cross_platform",
                    total_messages=total_messages,
                    processed_messages=total_messages,
                    successful_batches=1,
                    failed_batches=0,
                    total_tokens_used=response.token_count,
                    total_cost=0.0,  # Cost calculation moved to provider level
                    processing_time=processing_time,
                    summaries=[response.content],
                    errors=[]
                )
                
                self.logger.info(
                    f"跨平台分析完成: {result.total_tokens_used} tokens, "
                    f"耗时 {result.processing_time:.1f}s"
                )
                
                return result
            else:
                error_msg = f"LLM analysis failed: {response.error_message}"
                self.logger.error(error_msg)
                
                return BatchResult(
                    platform="cross_platform",
                    total_messages=total_messages,
                    processed_messages=0,
                    successful_batches=0,
                    failed_batches=1,
                    total_tokens_used=0,
                    total_cost=0.0,
                    processing_time=time.time() - start_time,
                    summaries=[],
                    errors=[error_msg]
                )
                
        except Exception as e:
            error_msg = f"Cross-platform analysis error: {str(e)}"
            self.logger.error(error_msg)
            
            return BatchResult(
                platform="cross_platform",
                total_messages=total_messages,
                processed_messages=0,
                successful_batches=0,
                failed_batches=1,
                total_tokens_used=0,
                total_cost=0.0,
                processing_time=time.time() - start_time,
                summaries=[],
                errors=[error_msg]
            )
    
    def _log_batch_execution_results(self, result: BatchResult) -> None:
        """
        记录批次执行结果的详细日志
        
        Args:
            result: 批次处理结果
        """
        # 基本执行信息
        self.logger.info(f"平台 {result.platform} 执行完成 - "
                        f"成功批次: {result.successful_batches}, "
                        f"失败批次: {result.failed_batches}, "
                        f"处理消息: {result.processed_messages}/{result.total_messages}")
        
        # 检查并记录有问题的批次
        problematic_batches = result.problematic_batches
        if problematic_batches:
            self.logger.warning(f"平台 {result.platform} 发现 {len(problematic_batches)} 个有问题的批次:")
            
            for batch_detail in problematic_batches:
                if not batch_detail.success:
                    # 完全失败的批次
                    self.logger.error(f"  批次 {batch_detail.batch_number} 执行失败 - "
                                    f"LLM命令: {batch_detail.llm_command}, "
                                    f"消息数: {batch_detail.message_count}, "
                                    f"错误: {batch_detail.error_message}")
                elif not batch_detail.has_meaningful_content:
                    # 成功但无有效内容的批次
                    self.logger.warning(f"  批次 {batch_detail.batch_number} 无有效结果 - "
                                      f"LLM命令: {batch_detail.llm_command}, "
                                      f"消息数: {batch_detail.message_count}, "
                                      f"Token使用: {batch_detail.tokens_used}, "
                                      f"处理时间: {batch_detail.processing_time:.2f}s")
                    # 记录响应内容摘要用于调试
                    content_preview = batch_detail.response_content[:200] + "..." if len(batch_detail.response_content) > 200 else batch_detail.response_content
                    self.logger.debug(f"    响应内容预览: {content_preview}")
        else:
            self.logger.info(f"平台 {result.platform} 所有批次执行正常，内容质量良好")
    
    def get_execution_summary(self, result: BatchResult) -> dict:
        """
        生成批次执行的汇总信息，用于报告生成
        
        Args:
            result: 批次处理结果
            
        Returns:
            包含执行汇总的字典
        """
        problematic_batches = result.problematic_batches
        failed_batches = result.failed_batch_details
        empty_batches = result.empty_result_batches
        
        return {
            'platform': result.platform,
            'total_batches': result.successful_batches + result.failed_batches,
            'successful_batches': result.successful_batches,
            'failed_batches': result.failed_batches,
            'problematic_count': len(problematic_batches),
            'failed_details': [
                {
                    'batch_number': detail.batch_number,
                    'command_type': detail.command_type,
                    'message_count': detail.message_count,
                    'error_message': detail.error_message,
                    'processing_time': detail.processing_time
                }
                for detail in failed_batches
            ],
            'empty_result_details': [
                {
                    'batch_number': detail.batch_number,
                    'command_type': detail.command_type,
                    'message_count': detail.message_count,
                    'tokens_used': detail.tokens_used,
                    'processing_time': detail.processing_time,
                    'response_preview': detail.response_content[:100] + "..." if len(detail.response_content) > 100 else detail.response_content
                }
                for detail in empty_batches
            ]
        }
