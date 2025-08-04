"""
Google Gemini LLM provider for TDXAgent.

This module provides integration with Google's Gemini API,
including Gemini Pro and other Gemini models.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from llm.base_provider import BaseLLMProvider, LLMRequest, LLMResponse
from utils.logger import TDXLogger


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider implementation.
    
    Features:
    - Support for Gemini Pro and other Gemini models
    - Accurate token counting
    - Comprehensive error handling
    - Rate limiting and retry logic
    - Support for system instructions
    """
    
    def __init__(self, config: Dict[str, Any], data_directory: Optional[str] = None):
        """
        Initialize Gemini provider.
        
        Args:
            config: Gemini configuration dictionary
        """
        super().__init__(config, "gemini", data_directory)
        
        # API configuration
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')
        
        # Model configurations
        self.model_configs = {
            'gemini-1.5-pro': {
                'max_tokens': 8192,
                'context_length': 2000000,  # 2M tokens
                'supports_system_instruction': True
            },
            'gemini-1.5-flash': {
                'max_tokens': 8192,
                'context_length': 1000000,  # 1M tokens
                'supports_system_instruction': True
            },
            'gemini-pro': {
                'max_tokens': 8192,
                'context_length': 32768,
                'supports_system_instruction': False
            },
            'gemini-pro-vision': {
                'max_tokens': 4096,
                'context_length': 16384,
                'supports_system_instruction': False,
                'supports_vision': True
            }
        }
        
        # Validate configuration
        if not self.api_key:
            self.logger.error("Gemini API key is required")
            raise ValueError("Gemini API key is required")
        
        self.logger.info(f"Initialized Gemini provider with base URL: {self.base_url}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        For Gemini, we use a simple estimation since tiktoken doesn't support Gemini.
        Google's tokenization is roughly similar to other models.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is an approximation and may not be perfectly accurate
        return max(1, len(text) // 4)
    
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Get the maximum context length for a model.
        
        Args:
            model: Model name (uses default if None)
            
        Returns:
            Maximum context length in tokens
        """
        model = model or self.default_model
        return self.model_configs.get(model, {}).get('context_length', 32768)
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]], 
                                          system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Args:
            messages: List of messages in OpenAI format
            system_prompt: System prompt (optional)
            
        Returns:
            Gemini-formatted request data
        """
        gemini_messages = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Convert role names
            if role == 'assistant':
                gemini_role = 'model'
            elif role == 'system':
                # System messages are handled separately in Gemini
                continue
            else:
                gemini_role = 'user'
            
            gemini_messages.append({
                'role': gemini_role,
                'parts': [{'text': content}]
            })
        
        request_data = {
            'contents': gemini_messages
        }
        
        # Add system instruction if supported and provided
        model = self.default_model
        if (system_prompt and 
            self.model_configs.get(model, {}).get('supports_system_instruction', False)):
            request_data['systemInstruction'] = {
                'parts': [{'text': system_prompt}]
            }
        
        return request_data
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request to the Gemini API.
        
        Args:
            request: LLM request object
            
        Returns:
            LLM response object
        """
        model = request.model or self.default_model
        
        # Construct URL
        url = f"{self.base_url}/models/{model}:generateContent"
        
        # Create call command string for logging
        call_command = f"API调用: Gemini {model} -> {url}"
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Convert messages to Gemini format
        request_data = self._convert_messages_to_gemini_format(
            request.messages, 
            request.system_prompt
        )
        
        # Add generation config
        generation_config = {}
        if request.max_tokens:
            generation_config['maxOutputTokens'] = request.max_tokens
        if request.temperature is not None:
            generation_config['temperature'] = request.temperature
        
        if generation_config:
            request_data['generationConfig'] = generation_config
        
        # Add safety settings (optional - can be configured)
        request_data['safetySettings'] = [
            {
                'category': 'HARM_CATEGORY_HARASSMENT',
                'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
            },
            {
                'category': 'HARM_CATEGORY_HATE_SPEECH',
                'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
            },
            {
                'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
            },
            {
                'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
            }
        ]
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Add API key to URL
                params = {'key': self.api_key}
                
                async with session.post(url, headers=headers, json=request_data, params=params) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        return self._parse_success_response(response_data, model, call_command, self.base_url)
                    else:
                        error_msg = self._parse_error_response(response_data, response.status)
                        return LLMResponse(
                            content="",
                            usage={},
                            model=model,
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
                model=model,
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
                model=model,
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
            # Extract content from candidates
            candidates = response_data.get('candidates', [])
            if not candidates:
                raise ValueError("No candidates in response")
            
            candidate = candidates[0]
            content_parts = candidate.get('content', {}).get('parts', [])
            
            if not content_parts:
                raise ValueError("No content parts in response")
            
            # Combine all text parts
            content = ""
            for part in content_parts:
                if 'text' in part:
                    content += part['text']
            
            # Extract usage information
            usage_metadata = response_data.get('usageMetadata', {})
            usage = {
                'prompt_tokens': usage_metadata.get('promptTokenCount', 0),
                'completion_tokens': usage_metadata.get('candidatesTokenCount', 0),
                'total_tokens': usage_metadata.get('totalTokenCount', 0)
            }
            
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
                base_url=self.base_url
            )
    
    def _parse_error_response(self, response_data: Dict[str, Any], status_code: int) -> str:
        """Parse an error response and return error message."""
        try:
            error_info = response_data.get('error', {})
            error_message = error_info.get('message', 'Unknown error occurred')
            error_code = error_info.get('code', status_code)
            
            # Handle specific error codes
            if status_code == 400:
                return f"Bad request: {error_message}"
            elif status_code == 401:
                return "Authentication failed. Please check your API key."
            elif status_code == 403:
                return "Access forbidden. Please check your API key permissions."
            elif status_code == 429:
                return "Rate limit exceeded. Please try again later."
            elif status_code == 500:
                return "Gemini server error. Please try again later."
            else:
                return f"API error ({error_code}): {error_message}"
                
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
            params = {'key': self.api_key}
            
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        for model in data.get('models', []):
                            model_name = model.get('name', '')
                            if model_name.startswith('models/'):
                                model_name = model_name[7:]  # Remove 'models/' prefix
                            if model_name:
                                models.append(model_name)
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
            'max_tokens': config.get('max_tokens', 8192),
            'context_length': config.get('context_length', 32768),
            'supports_system_instruction': config.get('supports_system_instruction', False),
            'supports_vision': config.get('supports_vision', False),
            'cost_per_1k_input_tokens': self._get_model_cost(model, 'input'),
            'cost_per_1k_output_tokens': self._get_model_cost(model, 'output')
        }
    
    def _get_model_cost(self, model: str, token_type: str) -> float:
        """Get the cost per 1K tokens for a model."""
        # Approximate costs for Gemini models (as of 2024)
        costs = {
            'gemini-1.5-pro': {'input': 0.0035, 'output': 0.0105},
            'gemini-1.5-flash': {'input': 0.00035, 'output': 0.00105},
            'gemini-pro': {'input': 0.0005, 'output': 0.0015},
            'gemini-pro-vision': {'input': 0.00025, 'output': 0.0005}
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
