"""
Claude CLI provider for TDXAgent.

This module provides integration with Claude Code CLI, using command-line interface
instead of API calls while maintaining the same interface as other providers.
"""

import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import subprocess

from llm.base_provider import BaseLLMProvider, LLMRequest, LLMResponse
from utils.logger import TDXLogger


class ClaudeProvider(BaseLLMProvider):
    """
    Claude CLI provider implementation.
    
    Features:
    - Command-line interface using 'claude' command
    - Unified interface with other providers
    - Automatic prompt file management
    - Error handling and retry logic
    - Markdown output format
    """
    
    def __init__(self, config: Dict[str, Any], data_directory: Optional[str] = None):
        """
        Initialize Claude CLI provider.
        
        Args:
            config: Claude CLI configuration dictionary
        """
        super().__init__(config, "claude_cli", data_directory)
        
        # CLI configuration
        self.cli_path = config.get('cli_path', 'claude')
        self.timeout = config.get('timeout', 120)  # 2 minutes default
        self.model = config.get('model', None)  # None means use default
        
        # Model configurations (for token estimation)
        self.model_configs = {
            'default': {
                'context_length': 200000,  # Conservative estimate
                'cost_per_1k_input': 0.003,  # Rough estimate
                'cost_per_1k_output': 0.015
            }
        }
        
        self.default_model = 'default'
        self.default_max_tokens = 4000
        
        self.logger = TDXLogger.get_logger("tdxagent.llm.claude_cli")
        
        # Validate CLI availability
        self._validate_cli_availability()
        
        self.logger.info("Initialized Claude CLI provider")
    
    def _validate_cli_availability(self) -> None:
        """Validate that Claude CLI is available and working."""
        try:
            result = subprocess.run(
                [self.cli_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version_info = result.stdout.strip()
                self.logger.info(f"Claude CLI available: {version_info}")
            else:
                raise RuntimeError(f"Claude CLI check failed: {result.stderr}")
                
        except FileNotFoundError:
            raise RuntimeError(f"Claude CLI not found at path: {self.cli_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI version check timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to validate Claude CLI: {e}")
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """
        Make a request to Claude CLI.
        
        Args:
            request: LLM request object
            
        Returns:
            LLM response object
        """
        # Convert request to single prompt
        if len(request.messages) == 1:
            prompt = request.messages[0]['content']
        else:
            # Combine messages (simple approach)
            prompt_parts = []
            for msg in request.messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt = '\n\n'.join(prompt_parts)
        
        return await self.generate_response(prompt, request.model)
    
    async def generate_response(self, prompt: str, model: Optional[str] = None, platform: Optional[str] = None) -> LLMResponse:
        """
        Generate response using Claude CLI.
        
        Args:
            prompt: Input prompt text
            model: Model name (ignored, uses CLI default)
            platform: Platform name for file naming (optional)
            
        Returns:
            LLMResponse with generated content
        """
        start_time = datetime.now()
        
        try:
            # Use the unified prompt file manager to save prompt
            if self._prompt_file_manager:
                analysis_type = "claude_cli"
                if platform:
                    analysis_type += f"_{platform}"
                analysis_type += "_analysis"
                prompt_file = self._prompt_file_manager.save_prompt(prompt, analysis_type)
            else:
                # Fallback to temporary file if prompt file manager not available
                import tempfile
                fd, prompt_file = tempfile.mkstemp(suffix='.txt', prefix='claude_prompt_')
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            
            # Build CLI command
            cmd_str = self._build_cli_command(prompt_file)
            
            self.logger.info(f"Executing Claude CLI command: {cmd_str}")
            self.logger.debug(f"Prompt file: {prompt_file}")
            self.logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Log prompt preview for debugging
            if len(prompt) > 200:
                self.logger.debug(f"Prompt preview (first 200 chars): {prompt[:200]}...")
            else:
                self.logger.debug(f"Prompt (full): {prompt}")
            
            # Execute command using shell
            process = await asyncio.create_subprocess_shell(
                cmd_str,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            try:
                exec_start = datetime.now()
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout
                )
                exec_time = (datetime.now() - exec_start).total_seconds()
                self.logger.debug(f"Claude CLI execution time: {exec_time:.2f}s")
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                self.logger.error(f"Claude CLI timed out after {self.timeout}s")
                raise RuntimeError(f"Claude CLI timed out after {self.timeout}s")
            
            # Cleanup fallback temporary file if used
            if not self._prompt_file_manager and (prompt_file.startswith('/tmp') or prompt_file.startswith('/var')):
                try:
                    os.unlink(prompt_file)
                except:
                    pass  # Ignore cleanup errors
            
            # Check result
            self.logger.debug(f"Claude CLI return code: {process.returncode}")
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8').strip() if stderr else ""
                stdout_msg = stdout.decode('utf-8').strip() if stdout else ""
                
                # Build comprehensive error message
                if error_msg:
                    full_error = f"Claude CLI stderr: {error_msg}"
                elif stdout_msg:
                    full_error = f"Claude CLI stdout (non-zero exit): {stdout_msg}"
                else:
                    full_error = f"Claude CLI exited with code {process.returncode} (no output)"
                
                # Add command info for debugging
                full_error += f" | Command: {cmd_str}"
                
                self.logger.error(f"Claude CLI failed: {full_error}")
                
                return LLMResponse(
                    content="",
                    usage={},
                    model=self.default_model,
                    provider=self.provider_name,
                    timestamp=start_time,
                    success=False,
                    error_message=f"CLI execution failed: {full_error}",
                    cost=0.0,
                    call_command=cmd_str,
                    base_url=None
                )
            
            # Parse output
            output = stdout.decode('utf-8').strip()
            
            # Debug logging for CLI output
            self.logger.debug(f"Claude CLI stdout length: {len(output)} characters")
            if len(output) > 500:
                self.logger.debug(f"Claude CLI stdout preview (first 500 chars): {output[:500]}...")
            else:
                self.logger.debug(f"Claude CLI stdout (full): {output}")
            
            if stderr:
                stderr_text = stderr.decode('utf-8').strip()
                if stderr_text:
                    self.logger.debug(f"Claude CLI stderr: {stderr_text}")
            
            if not output:
                self.logger.warning("Claude CLI returned empty output")
                return LLMResponse(
                    content="",
                    usage={},
                    model=self.default_model,
                    provider=self.provider_name,
                    timestamp=start_time,
                    success=False,
                    error_message="Empty response from Claude CLI",
                    cost=0.0,
                    call_command=cmd_str,
                    base_url=None
                )
            
            # Estimate tokens (rough calculation)
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(output)
            total_tokens = input_tokens + output_tokens
            
            # Create usage statistics
            usage = {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': total_tokens
            }
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate cost (Claude CLI usage varies, typically based on usage tiers)
            estimated_cost = 0.0  # Claude CLI 基础使用通常免费
            
            response = LLMResponse(
                content=output,
                usage=usage,
                model=self.default_model,
                provider=self.provider_name,
                timestamp=start_time,
                success=True,
                error_message=None,
                cost=estimated_cost,
                call_command=cmd_str,
                base_url=None  # Claude CLI doesn't use base URL
            )
            
            # Record conversation if enabled
            if self._collect_conversations:
                self._record_conversation(prompt, response, processing_time)
            
            self.logger.info(f"Claude CLI response: {total_tokens} tokens, {processing_time:.2f}s")
            self.logger.debug(f"Claude CLI response details: input={input_tokens}, output={output_tokens}, model={self.default_model}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Claude CLI generation failed: {e}")
            
            return LLMResponse(
                content="",
                usage={},
                model=self.default_model,
                provider=self.provider_name,
                timestamp=start_time,
                success=False,
                error_message=f"Generation error: {str(e)}",
                cost=0.0,
                call_command=getattr(locals(), 'cmd_str', 'command not available'),
                base_url=None
            )
    
    
    def _build_cli_command(self, prompt_file: str) -> str:
        """
        Build Claude CLI command string.
        
        Args:
            prompt_file: Path to prompt file
            
        Returns:
            Command string for shell execution
        """
        # Use -p flag for non-interactive mode with input redirection
        cmd_parts = [self.cli_path, '-p']
        
        # Add model if specified
        if self.model:
            cmd_parts.extend(['--model', self.model])
        
        # Use input redirection to avoid command line length limits
        cmd_str = ' '.join(cmd_parts) + f' < "{prompt_file}"'
        
        return cmd_str
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimation: ~4 characters per token for mixed content
        # This is conservative for Claude models
        return max(1, len(text) // 4)
    
    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """
        Calculate estimated cost for usage.
        
        Args:
            usage: Usage statistics
            
        Returns:
            Estimated cost in USD
        """
        if not usage:
            return 0.0
        
        config = self.model_configs.get(self.default_model, {})
        
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        input_cost = (input_tokens / 1000) * config.get('cost_per_1k_input', 0.003)
        output_cost = (output_tokens / 1000) * config.get('cost_per_1k_output', 0.015)
        
        return input_cost + output_cost
    
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Get context length for model.
        
        Args:
            model: Model name (ignored for CLI)
            
        Returns:
            Context length in tokens
        """
        return self.model_configs.get(self.default_model, {}).get('context_length', 200000)
    
    async def process_messages_batch(self, request: LLMRequest) -> LLMResponse:
        """
        Process batch request (delegates to generate_response).
        
        Args:
            request: LLM request object
            
        Returns:
            LLM response
        """
        # Convert messages to single prompt
        if len(request.messages) == 1:
            prompt = request.messages[0]['content']
        else:
            # Combine messages (simple approach)
            prompt_parts = []
            for msg in request.messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt = '\n\n'.join(prompt_parts)
        
        return await self.generate_response(prompt, request.model)
    
    async def health_check(self) -> bool:
        """
        Check if Claude CLI is available and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test with minimal prompt
            response = await self.generate_response(
                "Say 'OK' if you can respond.",
                max_tokens=10
            )
            return response.success and 'ok' in response.content.lower()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation."""
        return f"ClaudeProvider(cli_path={self.cli_path}, model={self.model})"