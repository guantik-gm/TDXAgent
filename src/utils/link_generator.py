"""
Link generation utilities for message references in TDXAgent.

This module provides functions to generate direct links to original messages
on various social media platforms for AI analysis citations.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import re


class LinkGenerator:
    """
    Generate direct links to original messages on social media platforms.
    
    Now supports unified formatting for platform-agnostic AI analysis.
    
    Supports:
    - Twitter/X: Direct tweet links
    - Telegram: Public channel/group links (when available)  
    - Discord: Message links (requires server membership)
    - Gmail: Direct email links
    - Unified: Platform-agnostic formatting for AI
    """
    
    def __init__(self):
        """Initialize link generator."""
        pass
    
    def generate_citation_reference(self, message: Dict[str, Any]) -> str:
        """
        Generate a citation reference for a message.
        
        Args:
            message: Message dictionary with platform and metadata
            
        Returns:
            Citation reference string (link or text description)
        """
        platform = message.get('platform', '').lower()
        
        if platform == 'twitter':
            return self._generate_twitter_reference(message)
        elif platform == 'telegram':
            return self._generate_telegram_reference(message)
        elif platform == 'discord':
            return self._generate_discord_reference(message)
        elif platform == 'gmail':
            return self._generate_gmail_reference(message)
        else:
            return self._generate_fallback_reference(message)
    
    def _generate_twitter_reference(self, message: Dict[str, Any]) -> str:
        """Generate Twitter tweet reference with direct link."""
        try:
            # 提取推文ID和用户名
            message_url = message.get('metadata', {}).get('message_url', '')
            author_name = message.get('author', {}).get('name', 'Unknown')
            
            if message_url and ('status/' in message_url or 'x.com' in message_url):
                # 直接使用消息URL，确保格式正确
                clean_url = message_url.replace('twitter.com', 'x.com')
                return f"[{author_name}的推文]({clean_url})"
            else:
                # 尝试从消息ID中提取推文ID构造链接
                message_id = message.get('id', '')
                if 'twitter_' in message_id:
                    tweet_id = message_id.split('twitter_')[-1]
                    # 尝试获取用户名
                    author_info = message.get('author', {})
                    username = author_info.get('username') or author_info.get('id', '')
                    
                    if username and tweet_id and username != 'Unknown':
                        # 清理用户名（移除@符号）
                        username = username.replace('@', '')
                        link = f"https://x.com/{username}/status/{tweet_id}"
                        return f"[{author_name}的推文]({link})"
                
                # Fallback to text description
                timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
                return f"{author_name} {timestamp}的推文"
                
        except Exception as e:
            # 调试信息
            print(f"DEBUG: Twitter link generation failed for {message.get('id', 'unknown')}: {e}")
            return self._generate_fallback_reference(message)
    
    def _generate_telegram_reference(self, message: Dict[str, Any]) -> str:
        """Generate Telegram message reference with text description and file location."""
        try:
            author_name = message.get('author', {}).get('name', 'Unknown')
            context = message.get('context', {})
            channel_name = context.get('channel', '')
            
            # 根据提示词模板要求，Telegram使用文本描述格式，不使用链接
            timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
            
            # 🎯 优化引用格式：使用平台缩写，简洁明了
            file_ref = message.get('_file_reference', {})
            platform_abbr = {
                'telegram': 'tg',
                'twitter': 'tw', 
                'gmail': 'gm',
                'discord': 'dc'
            }
            
            reference_info = ""
            if file_ref:
                line_number = file_ref.get('line_number')
                platform = message.get('platform', 'unknown')
                if line_number and platform:
                    abbr = platform_abbr.get(platform, platform[:2])
                    reference_info = f"[{abbr}:{line_number}] "
            
            if channel_name:
                # 清理群组名称
                clean_channel = channel_name.split('(')[0].strip() if '(' in channel_name else channel_name
                return f"{reference_info}{clean_channel} @{author_name} {timestamp}"
            else:
                return f"{reference_info}@{author_name} {timestamp}的Telegram消息"
                
        except Exception:
            return self._generate_fallback_reference(message)
    
    def _generate_discord_reference(self, message: Dict[str, Any]) -> str:
        """Generate Discord message reference."""
        try:
            author_name = message.get('author', {}).get('name', 'Unknown')
            context = message.get('context', {})
            channel_name = context.get('channel', '')
            server_name = context.get('server', '')
            
            # Discord链接格式：https://discord.com/channels/{server_id}/{channel_id}/{message_id}
            # 但需要用户是服务器成员才能访问，所以通常使用文本描述更实用
            
            timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
            
            if server_name and channel_name:
                return f"{server_name}#{channel_name} @{author_name} {timestamp}的消息"
            elif channel_name:
                return f"#{channel_name} @{author_name} {timestamp}的消息"
            else:
                return f"@{author_name} {timestamp}的Discord消息"
                
        except Exception:
            return self._generate_fallback_reference(message)
    
    def _generate_gmail_reference(self, message: Dict[str, Any]) -> str:
        """Generate Gmail message reference with direct link."""
        try:
            # 提取邮件信息
            author_name = message.get('author', {}).get('name', 'Unknown')
            author_email = message.get('author', {}).get('id', '')  # Gmail中ID就是邮箱地址
            message_url = message.get('metadata', {}).get('message_url', '')
            
            # 提取邮件主题
            platform_specific = message.get('metadata', {}).get('platform_specific', {})
            subject = platform_specific.get('subject', '')
            
            # 使用直接的Gmail链接
            if message_url and 'mail.google.com' in message_url:
                display_name = subject[:30] + '...' if len(subject) > 30 else subject
                if display_name:
                    return f"[{author_name}的邮件：{display_name}]({message_url})"
                else:
                    return f"[{author_name}的邮件]({message_url})"
            else:
                # Fallback to text description
                timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
                if subject:
                    subject_short = subject[:20] + '...' if len(subject) > 20 else subject
                    return f"{author_name} {timestamp}的邮件：{subject_short}"
                else:
                    return f"{author_name} {timestamp}的邮件"
                
        except Exception:
            return self._generate_fallback_reference(message)
    
    def _generate_fallback_reference(self, message: Dict[str, Any]) -> str:
        """Generate fallback text reference when link generation fails."""
        author_name = message.get('author', {}).get('name', 'Unknown')
        platform = message.get('platform', 'unknown').title()
        timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
        
        return f"@{author_name} {timestamp}的{platform}消息"
    
    def _format_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display."""
        if not timestamp_str:
            return "Unknown time"
        
        try:
            # 解析ISO格式时间戳
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime('%m-%d %H:%M')
        except:
            # 如果解析失败，返回原始字符串的前部分
            return timestamp_str[:16] if len(timestamp_str) > 16 else timestamp_str
    
    def format_telegram_messages_grouped(self, messages: list, batch_info: dict = None) -> str:
        """
        Format Telegram messages grouped by channel for efficient analysis.
        
        Args:
            messages: List of Telegram message dictionaries
            batch_info: Batch information for multi-batch scenarios
            
        Returns:
            Formatted string with messages grouped by channel with platform header
        """
        # 不添加引用格式说明 - 统一分析架构中由模板处理
        formatted_lines = []
        
        # 按群组分组消息
        groups = {}
        for message in messages:
            if not message.get('content', {}).get('text', '').strip():
                continue
                
            context = message.get('context', {})
            channel = context.get('channel', '未知群组')
            
            # 清理群组名称（移除警告文本）
            if '(' in channel:
                channel = channel.split('(')[0].strip()
            
            if channel not in groups:
                groups[channel] = []
            groups[channel].append(message)
        
        formatted_sections = []
        # 🎯 关键修复：使用全局消息序号，确保同一批次文件中序号唯一
        global_message_counter = 1  # 全局消息计数器，在同一批次文件中唯一
        
        for channel, channel_messages in groups.items():
            # 群组标题
            section_lines = [f"## {channel} 群组讨论"]
            
            # 🎯 按时间排序（解决时间递增但行号递减问题）
            # 先按时间排序，这样显示时时间是正确的递增顺序
            channel_messages.sort(key=lambda m: m.get('metadata', {}).get('posted_at', ''))
            
            for i, message in enumerate(channel_messages, 1):
                author = message.get('author', {}).get('name', 'Unknown')
                content = message.get('content', {}).get('text', '')
                timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
                
                # 验证引用完整性
                if not self._validate_telegram_reference_completeness(message, channel, author, timestamp):
                    print(f"WARNING: Telegram引用信息不完整 - {message.get('id', 'unknown')}")
                
                # 添加媒体指示符
                media_indicators = []
                if message.get('content', {}).get('media'):
                    media_indicators.append("📷")
                
                platform_specific = message.get('metadata', {}).get('platform_specific', {})
                if platform_specific.get('forward_from'):
                    media_indicators.append("➡️")
                
                media_str = " " + "".join(media_indicators) if media_indicators else ""
                
                # 🎯 关键修复：使用全局消息序号，确保同一批次文件中序号唯一
                # 例如：用户看到[tg1:5]，可以在提示词文件中搜索"[tg1:5"直接定位
                message_sequence_number = global_message_counter
                
                platform_abbr = {
                    'telegram': 'tg',
                    'twitter': 'tw', 
                    'gmail': 'gm',
                    'discord': 'dc'
                }
                
                # 生成提示词文件内行号引用格式（支持批次编号）
                platform = message.get('platform', 'unknown')
                abbr = platform_abbr.get(platform, platform[:2])
                
                # 批次编号逻辑：多批次用tg1、tg2，单批次用tg
                if batch_info and batch_info.get('total_batches', 1) > 1:
                    batch_num = batch_info.get('batch_number', 1)
                    batch_abbr = f"{abbr}{batch_num}"
                else:
                    batch_abbr = abbr
                
                # 格式：[平台缩写+批次:消息序号 时间] - 对应提示词文件中的消息顺序
                line_info = f"[{batch_abbr}:{message_sequence_number} {timestamp}]"
                
                # 格式：[平台:序号 时间] @用户: 内容  
                line = f"{i}. {line_info} @{author}: {content}{media_str}"
                section_lines.append(line)
                
                # 递增全局消息计数器
                global_message_counter += 1
            
            formatted_sections.append('\n'.join(section_lines))
        
        formatted_lines.extend(formatted_sections)
        return '\n\n'.join(formatted_lines)

    def format_messages_with_inline_links(self, messages: list, platform: str = None) -> str:
        """
        Format messages with inline citation links for LLM.
        
        Args:
            messages: List of message dictionaries
            platform: Platform name for header generation
            
        Returns:
            Formatted string with inline citation links and platform header
        """
        if not messages:
            return ""
        
        # 自动检测平台（如果未提供）
        if not platform and messages:
            platform = messages[0].get('platform', '').lower()
        
        # 不添加引用格式说明 - 统一分析架构中由模板处理
        
        formatted_lines = []
        
        for i, message in enumerate(messages, 1):
            # 基本消息信息
            author = message.get('author', {}).get('name', 'Unknown')
            content = message.get('content', {}).get('text', '')
            timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
            msg_platform = message.get('platform', '')
            
            # 跳过空消息
            if not content.strip():
                continue
            
            # 生成引用链接
            citation_ref = self.generate_citation_reference(message)
            
            # 验证引用完整性
            if not self._validate_reference_completeness(message, citation_ref):
                print(f"WARNING: {msg_platform}引用信息不完整 - {message.get('id', 'unknown')}")
            
            # 添加媒体指示符
            media_indicators = []
            if message.get('content', {}).get('media'):
                media_indicators.append("📷")
            
            # 检查转发信息
            platform_specific = message.get('metadata', {}).get('platform_specific', {})
            if msg_platform == 'twitter' and platform_specific.get('is_retweet'):
                media_indicators.append("🔄")
            elif msg_platform == 'telegram' and platform_specific.get('forward_from'):
                media_indicators.append("➡️")
            
            media_str = " " + "".join(media_indicators) if media_indicators else ""
            
            # 创建内联格式：[时间] [作者链接]: 内容 媒体标识
            if citation_ref.startswith('[') and '](' in citation_ref:
                # 可点击链接格式
                line = f"{i}. [{timestamp}] {citation_ref}: {content}{media_str}"
            else:
                # 文本描述格式
                line = f"{i}. [{timestamp}] {citation_ref}: {content}{media_str}"
            
            formatted_lines.append(line.strip())
        
        # 去重
        seen = set()
        unique_lines = []
        for line in formatted_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)

    def format_messages_with_references(self, messages: list) -> str:
        """
        Format messages for LLM with citation references.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted string with citation references for each message
        """
        formatted_lines = []
        references = {}  # 存储引用信息供AI使用
        
        for i, message in enumerate(messages, 1):
            # 基本消息信息
            author = message.get('author', {}).get('name', 'Unknown')
            content = message.get('content', {}).get('text', '')
            timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
            platform = message.get('platform', '')
            
            # 跳过空消息
            if not content.strip():
                continue
            
            # 生成引用
            citation_ref = self.generate_citation_reference(message)
            references[str(i)] = citation_ref
            
            # 添加上下文信息（频道/群组名称）
            channel_info = ""
            context = message.get('context', {})
            if platform == 'telegram' and context:
                channel = context.get('channel', '')
                if channel and len(channel) < 30:
                    channel_info = f" #{channel}"
            elif platform == 'discord' and context:
                channel = context.get('channel', '')
                if channel:
                    channel_info = f" #{channel}"
            elif platform == 'gmail':
                # 显示邮件主题作为上下文
                platform_specific = message.get('metadata', {}).get('platform_specific', {})
                subject = platform_specific.get('subject', '')
                if subject and len(subject) < 40:
                    channel_info = f" 📧{subject}"
            
            # 添加媒体指示符
            media_indicators = []
            if message.get('content', {}).get('media'):
                media_indicators.append("📷")
            
            # 检查转发信息
            platform_specific = message.get('metadata', {}).get('platform_specific', {})
            if platform == 'twitter' and platform_specific.get('is_retweet'):
                media_indicators.append("🔄")
            elif platform == 'telegram' and platform_specific.get('forward_from'):
                media_indicators.append("➡️")
            
            media_str = "".join(media_indicators)
            
            # 创建格式化行
            line = f"{i}. [{timestamp}] {author}{channel_info}: {content} {media_str}".strip()
            formatted_lines.append(line)
        
        # 去重
        seen = set()
        unique_lines = []
        for line in formatted_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        # 生成消息引用映射表
        reference_table = "\n\n📄 消息引用表（供分析使用）:\n"
        for i, message in enumerate(messages, 1):
            if message.get('content', {}).get('text', '').strip():
                citation_ref = self.generate_citation_reference(message)
                reference_table += f"消息{i}: {citation_ref}\n"
        
        # 返回格式化文本 + 引用表
        formatted_text = '\n'.join(unique_lines) + reference_table
        
        return formatted_text
    
    def _validate_reference_completeness(self, message: Dict[str, Any], citation_ref: str) -> bool:
        """
        验证引用信息的完整性。
        
        Args:
            message: 消息数据
            citation_ref: 生成的引用文本
            
        Returns:
            True if reference is complete, False otherwise
        """
        platform = message.get('platform', '').lower()
        
        if platform == 'twitter':
            return self._validate_twitter_reference_completeness(message, citation_ref)
        elif platform == 'telegram':
            context = message.get('context', {})
            channel = context.get('channel', '')
            author = message.get('author', {}).get('name', 'Unknown')
            timestamp = message.get('metadata', {}).get('posted_at', '')
            return self._validate_telegram_reference_completeness(message, channel, author, timestamp)
        elif platform == 'discord':
            return self._validate_discord_reference_completeness(message, citation_ref)
        elif platform == 'gmail':
            return self._validate_gmail_reference_completeness(message, citation_ref)
        else:
            return citation_ref and citation_ref != "Unknown"
    
    def _validate_twitter_reference_completeness(self, message: Dict[str, Any], citation_ref: str) -> bool:
        """验证Twitter引用完整性。"""
        # 检查是否有链接格式
        if citation_ref.startswith('[') and '](' in citation_ref and ('x.com' in citation_ref or 'twitter.com' in citation_ref):
            return True
        
        # 检查是否有必要的作者和时间信息
        author = message.get('author', {}).get('name', '')
        timestamp = message.get('metadata', {}).get('posted_at', '')
        
        return author and author != 'Unknown' and timestamp
    
    def _validate_telegram_reference_completeness(self, message: Dict[str, Any], channel: str, author: str, timestamp: str) -> bool:
        """验证Telegram引用完整性。"""
        # 检查必要信息是否完整
        if not channel or channel == '未知群组':
            return False
        if not author or author == 'Unknown':
            return False
        if not timestamp or timestamp == 'Unknown time':
            return False
        
        return True
    
    def _validate_discord_reference_completeness(self, message: Dict[str, Any], citation_ref: str) -> bool:
        """验证Discord引用完整性。"""
        context = message.get('context', {})
        channel = context.get('channel', '')
        server = context.get('server', '')
        author = message.get('author', {}).get('name', '')
        
        # 至少需要频道名和作者名
        return bool(channel and author and author != 'Unknown')
    
    def _validate_gmail_reference_completeness(self, message: Dict[str, Any], citation_ref: str) -> bool:
        """验证Gmail引用完整性。"""
        # 检查是否有邮件链接格式
        if citation_ref.startswith('[') and '](' in citation_ref and 'mailto:' in citation_ref:
            return True
        
        # 检查是否有必要的发件人和主题信息
        author = message.get('author', {}).get('name', '')
        platform_specific = message.get('metadata', {}).get('platform_specific', {})
        subject = platform_specific.get('subject', '')
        
        return author and author != 'Unknown' and subject
    
    # ========== 新增统一格式化方法 ==========
    
    def format_messages_unified(self, messages: list, enable_twitter_layering: bool = True, batch_info: dict = None) -> str:
        """
        统一格式化所有平台消息 - AI完全平台无关。
        自动按平台分组处理，应用各平台最优化的格式。
        
        Args:
            messages: List of message dictionaries from any platform
            enable_twitter_layering: Whether to enable Twitter timeline source layering
            batch_info: Batch information dict with keys like {'batch_number': 1, 'total_batches': 2, 'platform': 'telegram'}
            
        Returns:
            Formatted string with unified citation format
        """
        if not messages:
            return ""
        
        # 按平台分组消息
        platform_groups = {}
        for message in messages:
            platform = message.get('platform', 'unknown')
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(message)
        
        # 按平台使用优化的格式化方法
        formatted_sections = []
        
        for platform, platform_messages in platform_groups.items():
            if platform == 'telegram':
                # 使用Telegram群组优化格式，传递批次信息
                formatted_section = self.format_telegram_messages_grouped(platform_messages, batch_info)
            elif platform == 'twitter':
                # 使用Twitter分层或普通格式
                if enable_twitter_layering:
                    formatted_section = self.format_twitter_messages_layered(platform_messages)
                else:
                    formatted_section = self.format_messages_with_inline_links(platform_messages, 'twitter')
            elif platform == 'gmail':
                # 使用Gmail内联链接格式
                formatted_section = self.format_messages_with_inline_links(platform_messages, 'gmail')
            elif platform == 'discord':
                # 使用Discord内联链接格式
                formatted_section = self.format_messages_with_inline_links(platform_messages, 'discord')
            else:
                # 通用格式
                formatted_section = self._format_generic_messages(platform_messages)
            
            if formatted_section.strip():
                formatted_sections.append(formatted_section)
        
        return '\n\n'.join(formatted_sections)
    
    def format_twitter_messages_layered(self, messages: list) -> str:
        """
        Twitter分层格式化 - 根据timeline_source分为following和其他数据。
        Following数据优先展示，其他数据次要展示。
        
        Args:
            messages: List of Twitter message dictionaries
            
        Returns:
            Layered formatted string with <my_following_data> and <other_data> tags
        """
        if not messages:
            return ""
        
        # 根据timeline_source分层
        following_messages = []
        other_messages = []
        
        for message in messages:
            platform_specific = message.get('metadata', {}).get('platform_specific', {})
            timeline_source = platform_specific.get('timeline_source')
            
            if timeline_source == 'following':
                following_messages.append(message)
            else:
                # for_you, unknown或其他来源都归到other_data
                other_messages.append(message)
        
        formatted_sections = []
        
        # 优先处理Following数据（重点关注）
        if following_messages:
            following_formatted = self.format_messages_with_inline_links(following_messages, 'twitter')
            if following_formatted.strip():
                following_section = f"""<my_following_data>
⚠️ 该部分数据为我的重点关注推文，必须一个不漏地进行深度分析，优先级最高

{following_formatted.strip()}
</my_following_data>"""
                formatted_sections.append(following_section)
        
        # 处理其他数据（For You推荐等）
        if other_messages:
            other_formatted = self.format_messages_with_inline_links(other_messages, 'twitter')
            if other_formatted.strip():
                other_section = f"""<other_data>
以下为算法推荐和其他来源的推文数据，作为趋势补充参考，如果你认为其中有和 <my_following_data> 重点关注推文中类似的主题或者信息，请务必采取结合重点推文进行分析：

{other_formatted.strip()}
</other_data>"""
                formatted_sections.append(other_section)
        
        return '\n\n'.join(formatted_sections)
    
    def _format_generic_messages(self, messages: list) -> str:
        """
        通用格式化方法，用于未知平台或混合平台消息。
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted string
        """
        if not messages:
            return ""
        
        formatted_lines = []
        
        for i, message in enumerate(messages, 1):
            # 基本消息信息
            content = message.get('content', {}).get('text', '').strip()
            if not content:
                continue
            
            # 统一格式：序号. [时间] 引用信息: 内容 媒体标识
            timestamp = self._format_timestamp(message.get('metadata', {}).get('posted_at', ''))
            citation = self._generate_smart_citation(message)
            media_indicators = self._get_media_indicators(message)
            
            line = f"{i}. [{timestamp}] {citation}: {content}{media_indicators}"
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _generate_smart_citation(self, message: Dict[str, Any]) -> str:
        """
        智能生成引用 - 能点击的用链接，不能点击的用文本。
        
        Args:
            message: Message dictionary
            
        Returns:
            Smart citation string (clickable link or text description)
        """
        platform = message.get('platform', '').lower()
        
        # 尝试生成可点击链接
        clickable_link = self._try_clickable_link(message)
        if clickable_link:
            return clickable_link
        
        # 生成标准化文本描述
        return self._generate_text_citation(message)
    
    def _try_clickable_link(self, message: Dict[str, Any]) -> Optional[str]:
        """
        尝试生成可点击链接。
        
        Args:
            message: Message dictionary
            
        Returns:
            Clickable markdown link or None if not possible
        """
        platform = message.get('platform', '').lower()
        
        try:
            if platform == 'twitter':
                return self._try_twitter_link(message)
            elif platform == 'gmail':
                return self._try_gmail_link(message)
            # Telegram/Discord 通常无法生成通用可点击链接
            return None
        except Exception:
            return None
    
    def _try_twitter_link(self, message: Dict[str, Any]) -> Optional[str]:
        """尝试生成Twitter可点击链接。"""
        try:
            author_name = message.get('author', {}).get('name', 'Unknown')
            message_url = message.get('metadata', {}).get('message_url', '')
            
            if message_url and ('status/' in message_url or 'x.com' in message_url):
                clean_url = message_url.replace('twitter.com', 'x.com')
                return f"[{author_name}的推文]({clean_url})"
            
            # 尝试从消息ID构造链接
            message_id = message.get('id', '')
            if 'twitter_' in message_id:
                tweet_id = message_id.split('twitter_')[-1]
                author_info = message.get('author', {})
                username = author_info.get('username') or author_info.get('id', '')
                
                if username and tweet_id and username != 'Unknown':
                    username = username.replace('@', '')
                    link = f"https://x.com/{username}/status/{tweet_id}"
                    return f"[{author_name}的推文]({link})"
            
            return None
        except Exception:
            return None
    
    def _try_gmail_link(self, message: Dict[str, Any]) -> Optional[str]:
        """尝试生成Gmail可点击链接。"""
        try:
            author_name = message.get('author', {}).get('name', 'Unknown')
            message_url = message.get('metadata', {}).get('message_url', '')
            
            if message_url and 'mail.google.com' in message_url:
                platform_specific = message.get('metadata', {}).get('platform_specific', {})
                subject = platform_specific.get('subject', '')
                
                if subject:
                    display_name = subject[:30] + '...' if len(subject) > 30 else subject
                    return f"[{author_name}的邮件：{display_name}]({message_url})"
                else:
                    return f"[{author_name}的邮件]({message_url})"
            
            return None
        except Exception:
            return None
    
    def _generate_text_citation(self, message: Dict[str, Any]) -> str:
        """
        生成标准化文本引用。
        
        Args:
            message: Message dictionary
            
        Returns:
            Standardized text citation
        """
        author = message.get('author', {}).get('name', 'Unknown')
        platform = message.get('platform', '').title()
        context = self._get_context_info(message)
        
        if context:
            return f"{author}@{platform}({context})"
        else:
            return f"{author}@{platform}"
    
    def _get_context_info(self, message: Dict[str, Any]) -> str:
        """
        获取上下文信息（群组、频道、服务器等）。
        
        Args:
            message: Message dictionary
            
        Returns:
            Context information string
        """
        platform = message.get('platform', '').lower()
        context = message.get('context', {})
        
        if platform == 'telegram':
            channel = context.get('channel', '')
            if channel and channel != '未知群组':
                # 清理群组名称
                if '(' in channel:
                    channel = channel.split('(')[0].strip()
                return channel
        elif platform == 'discord':
            server = context.get('server', '')
            channel = context.get('channel', '')
            if server and channel:
                return f"{server}#{channel}"
            elif channel:
                return f"#{channel}"
        elif platform == 'gmail':
            platform_specific = message.get('metadata', {}).get('platform_specific', {})
            subject = platform_specific.get('subject', '')
            if subject:
                return subject[:20] + '...' if len(subject) > 20 else subject
        
        return ""
    
    def _get_media_indicators(self, message: Dict[str, Any]) -> str:
        """
        获取媒体指示符。
        
        Args:
            message: Message dictionary
            
        Returns:
            Media indicators string
        """
        indicators = []
        platform = message.get('platform', '').lower()
        
        # 媒体文件指示符
        if message.get('content', {}).get('media'):
            indicators.append("📷")
        
        # 转发/回复指示符
        platform_specific = message.get('metadata', {}).get('platform_specific', {})
        if platform == 'twitter' and platform_specific.get('is_retweet'):
            indicators.append("🔄")
        elif platform == 'telegram' and platform_specific.get('forward_from'):
            indicators.append("➡️")
        
        return " " + "".join(indicators) if indicators else ""