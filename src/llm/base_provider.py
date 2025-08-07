"""
Base LLM provider interface for TDXAgent.

This module provides the abstract base class that all LLM providers
inherit from, ensuring consistent interface and common functionality.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from utils.logger import TDXLogger, log_async_function_call
from utils.prompt_file_manager import PromptFileManager


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    usage: Dict[str, Any]
    model: str
    provider: str
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    cost: float = 0.0  # 估算的调用成本
    # 新增字段：具体的调用命令信息
    call_command: Optional[str] = None  # 具体的调用命令，如 "gemini -y < xxx_prompt.txt" 或 "API调用: OpenAI GPT-4"
    base_url: Optional[str] = None      # API调用的base_url
    
    @property
    def token_count(self) -> int:
        """Get total token count from usage."""
        return self.usage.get('total_tokens', 0)
    
    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get('prompt_tokens', 0) or self.usage.get('input_tokens', 0)
    
    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get('completion_tokens', 0) or self.usage.get('output_tokens', 0)


@dataclass
class LLMRequest:
    """Request to LLM provider."""
    messages: List[Dict[str, str]]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        request_dict = {
            'messages': self.messages,
            'model': self.model
        }
        
        if self.max_tokens is not None:
            request_dict['max_tokens'] = self.max_tokens
        if self.temperature is not None:
            request_dict['temperature'] = self.temperature
            
        return request_dict


@dataclass
class LLMConversation:
    """Complete LLM conversation record for debugging and analysis.
    
    This class captures complete request-response pairs at the LLM interface level,
    providing full transparency into what was sent to the AI model and what was
    returned. This replaces the old prompt collection approach and enables better
    debugging and model testing.
    
    Attributes:
        request_prompt: Complete prompt sent to the LLM
        response_content: Full response content from the LLM
        model: Model name used for the request
        tokens_used: Total tokens consumed (input + output)
        cost: Estimated cost of the request
        processing_time: Time taken to process the request
        timestamp: When the request was made
        success: Whether the request succeeded
        error_message: Error details if the request failed
    """
    request_prompt: str
    response_content: str
    model: str
    tokens_used: int
    cost: float
    processing_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for report generation.
        
        Returns:
            Dictionary with all conversation data for report appendix
        """
        return {
            'request_prompt': self.request_prompt,
            'response_content': self.response_content,
            'model': self.model,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error_message': self.error_message
        }


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class provides common functionality and enforces a consistent
    interface for all LLM providers (OpenAI, Gemini, etc.).
    
    Features:
    - Rate limiting and retry logic
    - Token counting and management
    - Error handling and logging
    - Usage tracking
    - Async support
    - Interface-level conversation collection for debugging
    
    The conversation collection system replaces the old prompt collection
    approach by automatically recording complete request-response pairs
    at the provider interface level, providing full transparency.
    """
    
    def __init__(self, config: Dict[str, Any], provider_name: str, data_directory: Optional[str] = None):
        """
        Initialize the base LLM provider.
        
        Args:
            config: Provider-specific configuration
            provider_name: Name of the provider (e.g., 'openai', 'gemini')
            data_directory: Data directory path for prompt files
        """
        self.config = config
        self.provider_name = provider_name
        self.logger = TDXLogger.get_logger(f"tdxagent.llm.{provider_name}")
        
        # Rate limiting
        self.max_requests_per_minute = config.get('max_requests_per_minute', 60)
        self.max_tokens_per_minute = config.get('max_tokens_per_minute', 100000)
        self.request_timeout = config.get('timeout', 30)
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # Usage tracking
        self._request_count = 0
        self._token_count = 0
        self._last_request_time = 0.0
        self._request_times: List[float] = []
        self._token_usage_times: List[tuple[float, int]] = []
        
        # Interface-level conversation collection for debugging and transparency
        self._collect_conversations = False
        self._collected_conversations: List[LLMConversation] = []
        
        # Unified prompt file management
        self._prompt_file_enabled = config.get('enable_prompt_files', True)
        self._prompt_file_manager = None
        if self._prompt_file_enabled:
            try:
                # Use configured data directory or default
                prompts_dir = "TDXAgent_Data/prompts"
                if data_directory:
                    prompts_dir = f"{data_directory}/prompts"
                self._prompt_file_manager = PromptFileManager(prompts_dir)
            except Exception as e:
                self.logger.warning(f"Failed to initialize prompt file manager: {e}")
                self._prompt_file_enabled = False
        
        # Model configuration
        self.default_model = config.get('model', '')
        self.default_max_tokens = config.get('max_tokens', 4000)
        self.default_temperature = config.get('temperature', 0.7)
        
        self.logger.info(f"Initialized {provider_name} LLM provider")
    
    @abstractmethod
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request to the LLM provider.
        
        Args:
            request: LLM request object
            
        Returns:
            LLM response object
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        pass
    
    @abstractmethod
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Get the maximum context length for a model.
        
        Args:
            model: Model name (uses default if None)
            
        Returns:
            Maximum context length in tokens
        """
        pass
    
    async def generate_response(self, prompt: str, 
                               system_prompt: Optional[str] = None,
                               model: Optional[str] = None,
                               max_tokens: Optional[int] = None,
                               temperature: Optional[float] = None,
                               platform: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            model: Model to use (optional)
            max_tokens: Maximum tokens to generate (optional)
            temperature: Temperature for generation (optional)
            platform: Platform name for file naming (optional)
            
        Returns:
            LLM response
        """
        # Save prompt to file if enabled
        if self._prompt_file_enabled and self._prompt_file_manager:
            try:
                # Construct full prompt for saving
                full_prompt = ""
                if system_prompt:
                    full_prompt += f"[SYSTEM]\n{system_prompt}\n\n"
                full_prompt += f"[USER]\n{prompt}"
                
                # Save prompt to file with platform information
                analysis_type = f"{self.provider_name}"
                if platform:
                    analysis_type += f"_{platform}"
                analysis_type += "_analysis"
                
                prompt_file_path = self._prompt_file_manager.save_prompt(
                    full_prompt, 
                    analysis_type=analysis_type
                )
                self.logger.debug(f"Saved prompt to: {prompt_file_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to save prompt file: {e}")
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Create request
        request = LLMRequest(
            messages=messages,
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature or self.default_temperature
        )
        
        return await self.make_request_with_retry(request)
    
    async def generate_summary(self, messages: List[Dict[str, Any]], 
                              prompt_template: str,
                              model: Optional[str] = None) -> LLMResponse:
        """
        Generate a summary of messages using a prompt template.
        
        Args:
            messages: List of message dictionaries
            prompt_template: Template with {data} placeholder
            model: Model to use (optional)
            
        Returns:
            LLM response with summary
        """
        # Format messages for prompt
        formatted_messages = self.format_messages_for_prompt(messages)
        
        # Create prompt from template
        prompt = prompt_template.format(data=formatted_messages)
        
        return await self.generate_response(prompt, model=model)
    
    def format_messages_for_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        格式化消息以供LLM处理 - 使用统一格式化方法，完全平台无关。
        
        Args:
            messages: 优化后的消息字典列表
            
        Returns:
            格式化的字符串表示
        """
        from utils.link_generator import LinkGenerator
        
        link_generator = LinkGenerator()
        return link_generator.format_messages_unified(messages)
    
    async def rate_limit_check(self, estimated_tokens: int = 0) -> None:
        """
        Check and enforce rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
        """
        current_time = time.time()
        
        # Clean old request times (older than 1 minute)
        cutoff_time = current_time - 60
        self._request_times = [t for t in self._request_times if t > cutoff_time]
        self._token_usage_times = [(t, tokens) for t, tokens in self._token_usage_times if t > cutoff_time]
        
        # Check request rate limit
        if len(self._request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self._request_times[0])
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Check token rate limit
        current_token_usage = sum(tokens for _, tokens in self._token_usage_times)
        if current_token_usage + estimated_tokens > self.max_tokens_per_minute:
            # Find when we can make the request
            required_tokens = estimated_tokens
            for t, tokens in sorted(self._token_usage_times):
                if current_token_usage - tokens + required_tokens <= self.max_tokens_per_minute:
                    wait_time = 60 - (current_time - t)
                    if wait_time > 0:
                        self.logger.info(f"Token rate limit reached, waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                    break
    
    async def make_request_with_retry(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request with retry logic and rate limiting.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        # Estimate tokens for rate limiting
        estimated_tokens = sum(self.estimate_tokens(msg['content']) for msg in request.messages)
        estimated_tokens += request.max_tokens or self.default_max_tokens
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                await self.rate_limit_check(estimated_tokens)
                
                # Make the request and record timing
                start_time = time.time()
                response = await self._make_request(request)
                processing_time = time.time() - start_time
                
                # Track usage
                current_time = time.time()
                self._request_times.append(current_time)
                self._token_usage_times.append((current_time, response.token_count))
                self._request_count += 1
                self._token_count += response.token_count
                
                # Record conversation if collection is enabled
                if self._collect_conversations:
                    # Reconstruct the full prompt from messages
                    full_prompt = ""
                    for msg in request.messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role == 'system':
                            full_prompt += f"[SYSTEM]\n{content}\n\n"
                        elif role == 'user':
                            full_prompt += f"[USER]\n{content}\n\n"
                    
                    self._record_conversation(full_prompt.strip(), response, processing_time)
                
                if response.success:
                    self.logger.debug(f"Request successful: {response.token_count} tokens")
                    return response
                else:
                    raise Exception(response.error_message or "Request failed")
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Request attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
        
        # Return failed response
        return LLMResponse(
            content="",
            usage={},
            model=request.model,
            provider=self.provider_name,
            timestamp=datetime.now(),
            success=False,
            error_message=str(last_exception)
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for this provider.
        
        Returns:
            Dictionary with usage statistics
        """
        current_time = time.time()
        
        # Calculate recent usage (last hour)
        hour_ago = current_time - 3600
        recent_requests = len([t for t in self._request_times if t > hour_ago])
        recent_tokens = sum(tokens for t, tokens in self._token_usage_times if t > hour_ago)
        
        return {
            'provider': self.provider_name,
            'total_requests': self._request_count,
            'total_tokens': self._token_count,
            'recent_requests_1h': recent_requests,
            'recent_tokens_1h': recent_tokens,
            'current_rpm': len([t for t in self._request_times if t > current_time - 60]),
            'current_tpm': sum(tokens for t, tokens in self._token_usage_times if t > current_time - 60),
            'max_rpm': self.max_requests_per_minute,
            'max_tpm': self.max_tokens_per_minute
        }
    
    @log_async_function_call(TDXLogger.get_logger("tdxagent.llm.base"))
    async def health_check(self) -> bool:
        """
        Perform a health check on the provider.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            response = await self.generate_response(
                "Hello, this is a health check. Please respond with 'OK'.",
                max_tokens=10
            )
            return response.success and 'ok' in response.content.lower()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def enable_conversation_collection(self) -> None:
        """Enable interface-level conversation collection for debugging.
        
        Activates automatic recording of all LLM requests and responses with
        complete metadata. This provides full transparency into what prompts
        are sent to the AI model and what responses are received.
        """
        self._collect_conversations = True
        self._collected_conversations.clear()
        self.logger.debug("LLM conversation collection enabled")
    
    def disable_conversation_collection(self) -> None:
        """Disable interface-level conversation collection.
        
        Stops recording LLM conversations and clears existing collection
        to free memory. Should be used when debugging is not needed.
        """
        self._collect_conversations = False
        self._collected_conversations.clear()
        self.logger.debug("LLM conversation collection disabled")
    
    def get_collected_conversations(self) -> List[LLMConversation]:
        """Get all collected LLM conversations.
        
        Returns a copy of all recorded conversations including both successful
        and failed requests. Each conversation contains complete request-response
        data with metadata for debugging and analysis.
        
        Returns:
            List of LLMConversation objects with complete conversation data
        """
        return self._collected_conversations.copy()
    
    def clear_collected_conversations(self) -> None:
        """Clear all collected conversations from memory.
        
        Removes all recorded conversations to free memory. Should be called
        after generating reports to prepare for the next analysis cycle.
        """
        self._collected_conversations.clear()
    
    def _record_conversation(self, prompt: str, response: LLMResponse, processing_time: float) -> None:
        """Record a complete conversation at the interface level.
        
        Automatically captures request-response pairs when collection is enabled.
        This method is called internally by make_request_with_retry and provides
        the foundation for the new conversation collection system.
        
        Args:
            prompt: Complete prompt sent to the LLM
            response: LLM response object with metadata
            processing_time: Time taken to process the request
        """
        if not self._collect_conversations:
            return
        
        # Calculate cost if the provider supports it
        cost = 0.0
        if hasattr(self, 'calculate_cost') and response.usage:
            try:
                cost = self.calculate_cost(response.usage, response.model)
            except Exception as e:
                self.logger.warning(f"Failed to calculate cost: {e}")
        
        conversation = LLMConversation(
            request_prompt=prompt,
            response_content=response.content if response.success else "",
            model=response.model,
            tokens_used=response.token_count,
            cost=cost,
            processing_time=processing_time,
            timestamp=response.timestamp,
            success=response.success,
            error_message=response.error_message
        )
        
        self._collected_conversations.append(conversation)
        self.logger.debug(f"Recorded LLM conversation: {len(prompt)} chars prompt, {len(response.content)} chars response")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"provider={self.provider_name}, "
                f"model={self.default_model})")
