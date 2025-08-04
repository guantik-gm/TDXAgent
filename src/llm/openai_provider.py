"""
OpenAI LLM provider for TDXAgent.

This module provides integration with OpenAI's API, including GPT models
and compatible APIs (like Azure OpenAI, local models, etc.).
"""

import asyncio
import aiohttp
import json
import tiktoken
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from llm.base_provider import BaseLLMProvider, LLMRequest, LLMResponse
from utils.logger import TDXLogger


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider implementation.
    
    Features:
    - Support for all OpenAI models (GPT-3.5, GPT-4, etc.)
    - Compatible with OpenAI-compatible APIs
    - Accurate token counting using tiktoken
    - Comprehensive error handling
    - Rate limiting and retry logic
    """
    
    def __init__(self, config: Dict[str, Any], data_directory: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            config: OpenAI configuration dictionary
        """
        super().__init__(config, "openai", data_directory)
        
        # API configuration
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.organization = config.get('organization', '')
        
        # Ensure base_url ends with /v1 if it's the standard OpenAI URL
        if self.base_url == 'https://api.openai.com' or self.base_url == 'https://api.openai.com/':
            self.base_url = 'https://api.openai.com/v1'
        
        # Model configurations
        self.model_configs = {
            'gpt-4': {'max_tokens': 8192, 'context_length': 8192},
            'gpt-4-32k': {'max_tokens': 32768, 'context_length': 32768},
            'gpt-4-turbo': {'max_tokens': 4096, 'context_length': 128000},
            'gpt-4o': {'max_tokens': 4096, 'context_length': 128000},
            'gpt-4o-mini': {'max_tokens': 16384, 'context_length': 128000},
            'gpt-3.5-turbo': {'max_tokens': 4096, 'context_length': 16385},
            'gpt-3.5-turbo-16k': {'max_tokens': 16384, 'context_length': 16385}
        }
        
        # Initialize tokenizer for the default model
        self._tokenizer = None
        self._init_tokenizer()
        
        # Validate configuration
        if not self.api_key:
            self.logger.error("OpenAI API key is required")
            raise ValueError("OpenAI API key is required")
        
        self.logger.info(f"Initialized OpenAI provider with base URL: {self.base_url}")
    
    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer for token counting."""
        try:
            # Try to get encoding for the default model
            model = self.default_model or 'gpt-3.5-turbo'
            
            # Map model names to encoding names
            encoding_map = {
                'gpt-4': 'cl100k_base',
                'gpt-4-32k': 'cl100k_base',
                'gpt-4-turbo': 'cl100k_base',
                'gpt-4o': 'cl100k_base',
                'gpt-4o-mini': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'gpt-3.5-turbo-16k': 'cl100k_base'
            }
            
            encoding_name = encoding_map.get(model, 'cl100k_base')
            self._tokenizer = tiktoken.get_encoding(encoding_name)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize tokenizer: {e}")
            # Fallback to a basic tokenizer
            self._tokenizer = None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text using tiktoken.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        try:
            if self._tokenizer:
                return len(self._tokenizer.encode(text))
            else:
                # Fallback estimation: roughly 4 characters per token
                return len(text) // 4
        except Exception as e:
            self.logger.warning(f"Token estimation failed: {e}")
            return len(text) // 4
    
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Get the maximum context length for a model.
        
        Args:
            model: Model name (uses default if None)
            
        Returns:
            Maximum context length in tokens
        """
        model = model or self.default_model
        return self.model_configs.get(model, {}).get('context_length', 4096)
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare headers for API requests."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        if self.organization:
            headers['OpenAI-Organization'] = self.organization
        
        return headers
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request to the OpenAI API.
        
        Args:
            request: LLM request object
            
        Returns:
            LLM response object
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._prepare_headers()
        
        # Create call command string for logging
        call_command = f"API调用: OpenAI {request.model} -> {self.base_url}/chat/completions"
        
        # Prepare request data
        data = request.to_dict()
        
        # Add additional parameters
        data.update({
            'stream': False,
            'user': 'tdxagent'
        })
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        return self._parse_success_response(response_data, request.model, call_command, self.base_url)
                    else:
                        error_msg = self._parse_error_response(response_data, response.status)
                        return LLMResponse(
                            content="",
                            usage={},
                            model=request.model,
                            provider=self.provider_name,
                            timestamp=datetime.now(),
                            success=False,
                            error_message=error_msg,
                            call_command=call_command,
                            base_url=self.base_url
                        )
        
        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {self.request_timeout}s"
            self.logger.error(error_msg)
            return LLMResponse(
                content="",
                usage={},
                model=request.model,
                provider=self.provider_name,
                timestamp=datetime.now(),
                success=False,
                error_message=error_msg,
                call_command=call_command,
                base_url=self.base_url
            )
        
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            self.logger.error(error_msg)
            return LLMResponse(
                content="",
                usage={},
                model=request.model,
                provider=self.provider_name,
                timestamp=datetime.now(),
                success=False,
                error_message=error_msg,
                call_command=call_command,
                base_url=self.base_url
            )
    
    def _parse_success_response(self, response_data: Dict[str, Any], model: str, call_command: str, base_url: str) -> LLMResponse:
        """Parse a successful API response."""
        try:
            # Extract content
            choices = response_data.get('choices', [])
            if not choices:
                raise ValueError("No choices in response")
            
            content = choices[0].get('message', {}).get('content', '')
            
            # Extract usage information
            usage = response_data.get('usage', {})
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                provider=self.provider_name,
                timestamp=datetime.now(),
                success=True,
                call_command=call_command,
                base_url=base_url
            )
            
        except Exception as e:
            error_msg = f"Failed to parse response: {str(e)}"
            self.logger.error(error_msg)
            return LLMResponse(
                content="",
                usage={},
                model=model,
                provider=self.provider_name,
                timestamp=datetime.now(),
                success=False,
                error_message=error_msg,
                call_command=call_command,
                base_url=base_url
            )
    
    def _parse_error_response(self, response_data: Dict[str, Any], status_code: int) -> str:
        """Parse an error response and return error message."""
        try:
            error_info = response_data.get('error', {})
            error_type = error_info.get('type', 'unknown_error')
            error_message = error_info.get('message', 'Unknown error occurred')
            
            # Handle specific error types
            if status_code == 401:
                return "Authentication failed. Please check your API key."
            elif status_code == 429:
                return "Rate limit exceeded. Please try again later."
            elif status_code == 400:
                return f"Bad request: {error_message}"
            elif status_code == 500:
                return "OpenAI server error. Please try again later."
            else:
                return f"API error ({status_code}): {error_message}"
                
        except Exception:
            return f"HTTP {status_code}: Failed to parse error response"
    
    async def validate_model(self, model: str) -> bool:
        """
        Validate if a model is available.
        
        Args:
            model: Model name to validate
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Try to make a simple request with the model
            test_request = LLMRequest(
                messages=[{"role": "user", "content": "test"}],
                model=model,
                max_tokens=1
            )
            
            response = await self._make_request(test_request)
            return response.success
            
        except Exception as e:
            self.logger.error(f"Model validation failed for {model}: {e}")
            return False
    
    async def list_available_models(self) -> List[str]:
        """
        List available models from the API.
        
        Returns:
            List of available model names
        """
        try:
            url = f"{self.base_url}/models"
            headers = self._prepare_headers()
            
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['id'] for model in data.get('data', [])]
                        return sorted(models)
                    else:
                        self.logger.error(f"Failed to list models: HTTP {response.status}")
                        return []
        
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model: Model name (uses default if None)
            
        Returns:
            Dictionary with model information
        """
        model = model or self.default_model
        config = self.model_configs.get(model, {})
        
        return {
            'name': model,
            'provider': self.provider_name,
            'max_tokens': config.get('max_tokens', 4096),
            'context_length': config.get('context_length', 4096),
            'supports_functions': model.startswith('gpt-4') or model.startswith('gpt-3.5-turbo'),
            'supports_vision': model in ['gpt-4-vision-preview', 'gpt-4o'],
            'cost_per_1k_input_tokens': self._get_model_cost(model, 'input'),
            'cost_per_1k_output_tokens': self._get_model_cost(model, 'output')
        }
    
    def _get_model_cost(self, model: str, token_type: str) -> float:
        """Get the cost per 1K tokens for a model."""
        # Approximate costs (as of 2024) - these should be updated regularly
        costs = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-32k': {'input': 0.06, 'output': 0.12},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004}
        }
        
        return costs.get(model, {}).get(token_type, 0.0)
    
    def calculate_cost(self, usage: Dict[str, Any], model: Optional[str] = None) -> float:
        """
        Calculate the cost of a request based on usage.
        
        Args:
            usage: Usage dictionary from API response
            model: Model name (uses default if None)
            
        Returns:
            Estimated cost in USD
        """
        model = model or self.default_model
        
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        input_cost = (input_tokens / 1000) * self._get_model_cost(model, 'input')
        output_cost = (output_tokens / 1000) * self._get_model_cost(model, 'output')
        
        return input_cost + output_cost
