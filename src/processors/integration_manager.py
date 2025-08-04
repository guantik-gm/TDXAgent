"""
Integration Manager for TDXAgent progressive batch analysis.

This module provides intelligent context management and compression
for progressive batch analysis, ensuring cohesive final reports.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from utils.logger import TDXLogger


@dataclass
class AnalysisContext:
    """Container for compressed analysis context."""
    key_topics: List[str]
    important_conclusions: List[str]
    citations: List[str]
    summary: str
    token_count: int
    
    def to_compressed_text(self) -> str:
        """Convert to compressed text format for AI processing."""
        sections = []
        
        if self.key_topics:
            sections.append("主要话题：" + "；".join(self.key_topics))
        
        if self.important_conclusions:
            sections.append("重要结论：" + "；".join(self.important_conclusions))
        
        if self.citations:
            sections.append("已分析消息引用：" + "；".join(self.citations))
        
        return " | ".join(sections)


class IntegrationManager:
    """
    Intelligent integration manager for progressive batch analysis.
    
    Features:
    - Smart context compression to fit token limits
    - Citation reference preservation
    - Key information extraction and synthesis
    - Progressive analysis context management
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize integration manager.
        
        Args:
            llm_provider: LLM provider for token estimation
        """
        self.llm_provider = llm_provider
        self.logger = TDXLogger.get_logger("tdxagent.processors.integration")
        
        # Compression settings
        self.max_context_ratio = 0.3  # 最多使用30%的token用于上下文
        self.min_context_tokens = 500  # 最小上下文token数
        self.max_context_tokens = 8000  # 最大上下文token数
        
        self.logger.info("Initialized integration manager")
    
    def compress_analysis_context(self, full_analysis: str, max_tokens: int) -> AnalysisContext:
        """
        Compress full analysis into key context for next batch.
        
        Args:
            full_analysis: Complete analysis text from previous batch
            max_tokens: Maximum tokens allowed for compressed context
            
        Returns:
            AnalysisContext with compressed information
        """
        try:
            # Extract key components
            key_topics = self._extract_key_topics(full_analysis)
            important_conclusions = self._extract_important_conclusions(full_analysis)
            citations = self._extract_citations(full_analysis)
            
            # Create initial context
            context = AnalysisContext(
                key_topics=key_topics,
                important_conclusions=important_conclusions,
                citations=citations,
                summary="",
                token_count=0
            )
            
            # Generate compressed text and check token count
            compressed_text = context.to_compressed_text()
            token_count = self._estimate_tokens(compressed_text)
            
            # If too long, progressively reduce content
            if token_count > max_tokens:
                context = self._progressive_compression(context, max_tokens)
                compressed_text = context.to_compressed_text()
                token_count = self._estimate_tokens(compressed_text)
            
            # If compressed text is empty, use truncated original analysis as fallback
            if not compressed_text.strip():
                self.logger.warning("Information extraction failed, using truncated original analysis as fallback")
                # Truncate original analysis to fit token limit
                words = full_analysis.split()
                max_words = max_tokens * 3  # Rough estimate: 3 words per token
                if len(words) > max_words:
                    truncated_analysis = " ".join(words[:max_words]) + "..."
                else:
                    truncated_analysis = full_analysis
                
                context.summary = truncated_analysis
                context.token_count = self._estimate_tokens(truncated_analysis)
            else:
                context.summary = compressed_text
                context.token_count = token_count
            
            self.logger.debug(f"Compressed analysis: {context.token_count} tokens, "
                            f"{len(key_topics)} topics, {len(important_conclusions)} conclusions")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to compress analysis context: {e}")
            # Return minimal context on error
            return AnalysisContext(
                key_topics=[],
                important_conclusions=[],
                citations=[],
                summary="前一批次分析出现处理错误",
                token_count=self._estimate_tokens("前一批次分析出现处理错误")
            )
    
    def _extract_key_topics(self, analysis: str) -> List[str]:
        """Extract key topics from analysis text."""
        topics = []
        
        # Pattern 1: 关键话题/主要话题/讨论要点 sections
        topic_patterns = [
            r'##\s*(?:关键话题|主要话题|讨论要点|主要讨论话题)[\s\S]*?(?=##|$)',
            r'###\s*(?:关键话题|主要话题|讨论要点|主要讨论话题)[\s\S]*?(?=###|##|$)'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, analysis, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Extract numbered items
                items = re.findall(r'\d+\.\s*([^(]+)(?:\([^)]*\))?', match)
                for item in items:
                    topic = item.strip().rstrip('：:')
                    if len(topic) > 10 and len(topic) < 100:  # Reasonable length
                        topics.append(topic)
        
        # Limit to most important topics
        return topics[:8]
    
    def _extract_important_conclusions(self, analysis: str) -> List[str]:
        """Extract important conclusions from analysis text."""
        conclusions = []
        
        # Pattern for important information sections
        conclusion_patterns = [
            r'##\s*(?:重要信息|重要信息摘要|重要结论|值得关注)[\s\S]*?(?=##|$)',
            r'###\s*(?:重要信息|重要信息摘要|重要结论|值得关注)[\s\S]*?(?=###|##|$)'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, analysis, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Extract bullet points and numbered items
                items = re.findall(r'[-•]\s*([^(]+)(?:\([^)]*\))?', match)
                items.extend(re.findall(r'\d+\.\s*([^(]+)(?:\([^)]*\))?', match))
                
                for item in items:
                    conclusion = item.strip().rstrip('：:')
                    if len(conclusion) > 10 and len(conclusion) < 150:
                        conclusions.append(conclusion)
        
        # Limit to most important conclusions
        return conclusions[:10]
    
    def _extract_citations(self, analysis: str) -> List[str]:
        """Extract citation references from analysis text."""
        citations = []
        
        # Extract various citation formats
        citation_patterns = [
            r'\[([^]]+的推文)\]\([^)]+\)',  # Twitter links
            r'\[([^]]+的邮件[^]]*)\]\([^)]+\)',  # Gmail links
            r'@(\w+)\s+\d{2}-\d{2}\s+\d{2}:\d{2}的(?:推文|消息|邮件)',  # Text citations
            r'消息(\d+)',  # Message numbers
            r'基于[：:](?:消息)?(\d+(?:,\d+)*)',  # Based on message numbers
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, analysis)
            for match in matches:
                if isinstance(match, tuple):
                    citations.extend(match)
                else:
                    citations.append(str(match))
        
        # Remove duplicates and filter
        unique_citations = []
        seen = set()
        for citation in citations:
            citation_clean = citation.strip()
            if citation_clean and citation_clean not in seen and len(citation_clean) < 50:
                seen.add(citation_clean)
                unique_citations.append(citation_clean)
        
        return unique_citations[:15]  # Limit citations
    
    def _progressive_compression(self, context: AnalysisContext, max_tokens: int) -> AnalysisContext:
        """Progressively compress context to fit token limit."""
        # Step 1: Reduce citations
        if len(context.citations) > 10:
            context.citations = context.citations[:10]
        
        # Step 2: Reduce conclusions
        if len(context.important_conclusions) > 6:
            context.important_conclusions = context.important_conclusions[:6]
        
        # Step 3: Reduce topics
        if len(context.key_topics) > 5:
            context.key_topics = context.key_topics[:5]
        
        # Step 4: Truncate individual items if still too long
        compressed_text = context.to_compressed_text()
        if self._estimate_tokens(compressed_text) > max_tokens:
            # Truncate topics
            context.key_topics = [topic[:30] + "..." if len(topic) > 30 else topic 
                                for topic in context.key_topics]
            
            # Truncate conclusions
            context.important_conclusions = [
                conclusion[:50] + "..." if len(conclusion) > 50 else conclusion 
                for conclusion in context.important_conclusions
            ]
        
        return context
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.llm_provider and hasattr(self.llm_provider, 'estimate_tokens'):
            return self.llm_provider.estimate_tokens(text)
        else:
            # Rough estimation: ~4 characters per token for Chinese text
            return len(text) // 3
    
    def calculate_context_token_limit(self, total_batch_tokens: int, buffer_tokens: int) -> int:
        """
        Calculate appropriate token limit for context compression.
        
        Args:
            total_batch_tokens: Total tokens available for this batch
            buffer_tokens: Tokens reserved for response
            
        Returns:
            Maximum tokens to use for previous context
        """
        available_tokens = total_batch_tokens - buffer_tokens
        
        # Use at most 30% for context, but within min/max bounds
        context_tokens = int(available_tokens * self.max_context_ratio)
        context_tokens = max(self.min_context_tokens, context_tokens)
        context_tokens = min(self.max_context_tokens, context_tokens)
        
        # Ensure we don't exceed available tokens
        context_tokens = min(context_tokens, available_tokens // 2)
        
        self.logger.debug(f"Context token limit: {context_tokens} "
                         f"(available: {available_tokens}, ratio: {self.max_context_ratio})")
        
        return context_tokens
    
    def merge_citation_references(self, old_refs: List[str], new_refs: List[str]) -> List[str]:
        """
        Merge citation references from previous and current batch.
        
        Args:
            old_refs: References from previous analysis
            new_refs: References from current analysis
            
        Returns:
            Merged list of unique references
        """
        merged = []
        seen = set()
        
        # Add old references first (maintain order)
        for ref in old_refs:
            if ref not in seen:
                seen.add(ref)
                merged.append(ref)
        
        # Add new references
        for ref in new_refs:
            if ref not in seen:
                seen.add(ref)
                merged.append(ref)
        
        return merged
    
    def should_use_integration_mode(self, batch_number: int, total_batches: int) -> bool:
        """
        Determine if integration mode should be used for this batch.
        
        Args:
            batch_number: Current batch number (1-based)
            total_batches: Total number of batches
            
        Returns:
            True if integration mode should be used
        """
        # Always use integration mode for batch 2 and beyond
        return batch_number > 1
    
    def create_integration_prompt_context(self, previous_analysis: str, 
                                        platform: str = "") -> str:
        """
        Create context text for integration prompt.
        
        Args:
            previous_analysis: Complete analysis text from previous batch
            platform: Platform name for context
            
        Returns:
            Formatted context text for prompt
        """
        if not previous_analysis or not previous_analysis.strip():
            return ""
        
        context_lines = [
            "【前批次分析结果】",
            previous_analysis,
            "",
            "【整合要求】",
            "请将新数据的分析与上述前批次结果进行整合，形成统一连贯的完整分析。",
            "- 避免重复已分析的内容",
            "- 保持引用链接的完整性",
            "- 确保最终结果具有整体性和连贯性",
            "- 如有新的重要发现，请突出说明",
            ""
        ]
        
        return "\n".join(context_lines)
    
    def validate_integration_result(self, result: str) -> Tuple[bool, str]:
        """
        Validate the integration analysis result.
        
        Args:
            result: Integration analysis result
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not result or len(result.strip()) < 50:
            return False, "Integration result is too short"
        
        # Check for basic structure
        if not re.search(r'##\s*\w+', result):
            return False, "Integration result lacks proper structure"
        
        # Check for citations (should have some references)
        has_citations = bool(re.search(r'\[[^\]]+\]\([^)]+\)|消息\d+|基于[：:]', result))
        if not has_citations:
            return False, "Integration result lacks citation references"
        
        return True, ""
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get statistics about compression operations."""
        return {
            'max_context_ratio': self.max_context_ratio,
            'min_context_tokens': self.min_context_tokens,
            'max_context_tokens': self.max_context_tokens,
            'compression_enabled': True
        }