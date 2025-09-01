"""
优化版消息格式转换器 - 只保留核心有用字段。

这个模块将平台特定的消息数据转换为精简的TDXAgent标准格式，
大幅减少存储空间和token消耗，只保留最有价值的信息。
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from utils.logger import TDXLogger
from utils.helpers import generate_message_id


class OptimizedMessageConverter:
    """
    优化版统一消息格式转换器。
    
    转换为精简的TDXAgent格式：
    {
        "id": "unique_message_id",
        "platform": "twitter|telegram|discord", 
        "author": {
            "name": "用户名",
            "id": "用户ID",
            "username": "用户名（Twitter/Telegram适用）"
        },
        "content": {
            "text": "消息内容",
            "media": ["媒体URL数组"] // 仅当有实际媒体时才包含
        },
        "metadata": {
            "posted_at": "ISO时间戳",
            "message_url": "原始消息链接", // 仅Twitter有效，其他平台省略
            "platform_specific": {
                // 平台特定的核心数据（如互动数据、消息ID等）
            }
        },
        "context": {
            "channel": "频道/群组名称", // 仅Telegram/Discord适用
            "group": "群组名称" // 仅Telegram适用
        }
    }
    """
    
    def __init__(self):
        self.logger = TDXLogger.get_logger("tdxagent.storage.optimized_converter")
        
        # 平台特定转换器
        self.converters = {
            'twitter': self._convert_twitter_message,
            'telegram': self._convert_telegram_message,
            'discord': self._convert_discord_message
        }
    
    def convert_message(self, raw_message: Dict[str, Any], 
                       platform: str) -> Optional[Dict[str, Any]]:
        """
        将平台特定消息转换为优化的统一格式。
        
        Args:
            raw_message: 平台原始消息数据
            platform: 平台名称
            
        Returns:
            转换后的优化消息格式，失败返回None
        """
        try:
            platform_lower = platform.lower()
            
            if platform_lower not in self.converters:
                self.logger.error(f"不支持的平台: {platform}")
                return None
            
            # 使用平台特定转换器
            converter = self.converters[platform_lower]
            converted_message = converter(raw_message)
            
            if not converted_message:
                self.logger.warning(f"转换 {platform} 消息失败")
                return None
            
            # 基础验证
            if not self._validate_optimized_message(converted_message):
                self.logger.warning(f"转换后的 {platform} 消息验证失败")
                return None
            
            self.logger.debug(f"成功转换 {platform} 消息: {converted_message.get('id')}")
            return converted_message
            
        except Exception as e:
            self.logger.error(f"转换 {platform} 消息时出错: {e}")
            return None
    
    def _convert_twitter_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """转换Twitter消息为优化格式。"""
        try:
            # 提取基本消息信息
            tweet_id = raw_message.get('id')
            if not tweet_id:
                return None
            
            # 生成统一ID
            unified_id = generate_message_id('twitter', str(tweet_id))
            
            # 作者信息（精简）
            author_data = raw_message.get('author', {})
            author = {
                "name": author_data.get('name', 'Unknown'),
                "id": author_data.get('id', 'unknown')
            }
            
            # 如果有用户名，则添加
            username = author_data.get('username')
            if username and username != author['name']:
                author['username'] = username
            
            # 内容信息（精简）
            content = {
                "text": raw_message.get('text', '')
            }
            
            # 只有实际媒体时才添加media字段
            media_urls = raw_message.get('media_urls', [])
            if media_urls:
                # 过滤掉头像URL，只保留实际内容媒体
                actual_media = [url for url in media_urls if self._is_actual_media(url)]
                if actual_media:
                    content['media'] = actual_media
            
            # 元数据（核心信息）- 版本化时区处理
            raw_created_at = raw_message.get('created_at', '')
            
            # 智能时区处理
            if isinstance(raw_created_at, datetime):
                # datetime对象，转换为本地时区
                if raw_created_at.tzinfo:
                    local_time = raw_created_at.astimezone()
                else:
                    local_time = raw_created_at
                posted_at = local_time.isoformat()
                timezone_info = str(local_time.tzinfo) if local_time.tzinfo else "本地时区"
            else:
                # 字符串格式，可能需要解析
                try:
                    dt = datetime.fromisoformat(str(raw_created_at).replace('Z', '+00:00'))
                    if dt.tzinfo:
                        local_time = dt.astimezone()
                        posted_at = local_time.isoformat()
                        timezone_info = str(local_time.tzinfo)
                    else:
                        posted_at = str(raw_created_at)
                        timezone_info = "inherited"
                except:
                    posted_at = str(raw_created_at)
                    timezone_info = "inherited"
            
            metadata = {
                "posted_at": posted_at,
                "platform_specific": {
                    # 🎯 版本标识
                    "timezone_version": "v2.0_local",
                    "timezone_info": timezone_info
                }
            }
            
            # 添加消息URL（Twitter特有）
            if raw_message.get('url'):
                # 优先使用原始消息中的URL
                metadata['message_url'] = raw_message['url']
            elif author['name'] != 'Unknown' and tweet_id:
                # 备用：构造URL
                username = author.get('username', author['name'])
                metadata['message_url'] = f"https://x.com/{username}/status/{tweet_id}"
            
            # 保留互动数据（如果有意义的数值）
            engagement = raw_message.get('engagement', {})
            if engagement:
                platform_data = {}
                for key in ['likes', 'retweets', 'replies', 'quotes']:
                    value = engagement.get(key, 0)
                    if value > 0:  # 只保存有数值的字段
                        platform_data[f"{key}_count"] = value
                
                if platform_data:
                    metadata['platform_specific'] = platform_data
            
            # 添加转发标识（如果是转发）
            if raw_message.get('is_retweet'):
                if 'platform_specific' not in metadata:
                    metadata['platform_specific'] = {}
                metadata['platform_specific']['is_retweet'] = True
            
            # 添加timeline来源标识
            timeline_source = raw_message.get('timeline_source')
            if timeline_source and timeline_source != 'unknown':
                if 'platform_specific' not in metadata:
                    metadata['platform_specific'] = {}
                metadata['platform_specific']['timeline_source'] = timeline_source
            
            # 上下文（Twitter不需要频道信息，但验证器需要这个字段）
            result = {
                'id': unified_id,
                'platform': 'twitter',
                'author': author,
                'content': content,
                'metadata': metadata,
                'context': {}  # 空的上下文，满足验证器要求
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"转换Twitter消息时出错: {e}")
            return None
    
    def _convert_telegram_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """转换Telegram消息为优化格式。"""
        try:
            # 提取基本消息信息
            message_id = raw_message.get('id')
            if not message_id:
                return None
            
            # 生成统一ID
            unified_id = generate_message_id('telegram', str(message_id))
            
            # 作者信息（精简）- 处理各种消息类型
            from_user = raw_message.get('from_user') or raw_message.get('sender', {})
            
            # 如果没有from_user，可能是频道消息或系统消息
            if not from_user:
                chat_info = raw_message.get('chat', {})
                if chat_info.get('title'):
                    # 使用频道/群组信息作为作者
                    from_user = {
                        'first_name': chat_info['title'],
                        'id': chat_info.get('id', 0),
                        'username': chat_info.get('username', '')
                    }
                else:
                    self.logger.debug(f"Telegram消息无作者信息: {message_id}")
                    return None
            
            # 构建作者名称
            first_name = from_user.get('first_name', '') or ''
            last_name = from_user.get('last_name', '') or ''
            full_name = (first_name + ' ' + last_name).strip()
            
            # 确保必要字段存在
            author_name = full_name or from_user.get('username', '') or 'Unknown'
            author_id = str(from_user.get('id', '') or '')
            
            if not author_id:
                self.logger.debug(f"Telegram消息作者ID为空: {message_id}, from_user: {from_user}")
                return None
            
            author = {
                "name": author_name,
                "id": author_id
            }
            
            # 如果username与name不同，则添加username字段
            username = from_user.get('username', '')
            if username and username != author_name:
                author['username'] = username
            
            # 内容信息（精简）
            content = {
                "text": raw_message.get('text', '') or raw_message.get('message', '')
            }
            
            # 检查媒体文件
            media_files = []
            for media_type in ['photo', 'video', 'document', 'voice', 'video_note']:
                if raw_message.get(media_type):
                    media_files.append(raw_message[media_type])
            
            if media_files:
                content['media'] = media_files
            
            # 元数据（核心信息）- 版本化时区处理
            raw_date = raw_message.get('date', '') or raw_message.get('timestamp', '') or raw_message.get('created_at', '')
            if not raw_date:
                self.logger.debug(f"Telegram消息缺少时间戳: {message_id}")
                return None
            
            # 智能时区处理
            if isinstance(raw_date, datetime):
                # Telethon返回的datetime对象，转换为本地时区
                if raw_date.tzinfo:
                    local_time = raw_date.astimezone()
                else:
                    local_time = raw_date
                posted_at = local_time.isoformat()
                timezone_info = str(local_time.tzinfo) if local_time.tzinfo else "本地时区"
            else:
                # 字符串格式，直接使用
                posted_at = str(raw_date)
                timezone_info = "inherited"
                
            metadata = {
                "posted_at": posted_at,
                "platform_specific": {
                    "message_id": message_id,
                    "chat_id": raw_message.get('chat', {}).get('id', ''),
                    # 🎯 版本标识
                    "timezone_version": "v2.0_local",
                    "timezone_info": timezone_info
                }
            }
            
            # 转发信息（如果有）
            forward_from = raw_message.get('forward_from')
            if forward_from:
                metadata['platform_specific']['forward_from'] = forward_from
            
            # 上下文（频道/群组信息）
            chat_info = raw_message.get('chat', {})
            context = {}
            
            chat_title = chat_info.get('title', '')
            if chat_title:
                context['channel'] = chat_title
                # 如果是群组类型，也设置group字段
                if chat_info.get('type') in ['group', 'supergroup']:
                    context['group'] = chat_title
            
            result = {
                'id': unified_id,
                'platform': 'telegram',
                'author': author,
                'content': content,
                'metadata': metadata,
                'context': context  # 总是包含context字段（可能为空），满足验证器要求
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"转换Telegram消息时出错: {e}")
            return None
    
    def _convert_discord_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """转换Discord消息为优化格式。"""
        try:
            # 提取基本消息信息
            message_id = raw_message.get('id')
            if not message_id:
                return None
            
            # 生成统一ID
            unified_id = generate_message_id('discord', str(message_id))
            
            # 作者信息（精简）
            author_data = raw_message.get('author', {})
            author = {
                "name": author_data.get('username', 'Unknown'),
                "id": str(author_data.get('id', ''))
            }
            
            # 内容信息（精简）
            content = {
                "text": raw_message.get('content', '')
            }
            
            # 处理附件和媒体
            media_urls = []
            
            # 处理attachments
            attachments = raw_message.get('attachments', [])
            for attachment in attachments:
                if isinstance(attachment, dict):
                    url = attachment.get('url') or attachment.get('proxy_url')
                    if url:
                        media_urls.append(url)
            
            # 处理embeds中的媒体
            embeds = raw_message.get('embeds', [])
            for embed in embeds:
                if isinstance(embed, dict):
                    for media_key in ['image', 'video', 'thumbnail']:
                        media_obj = embed.get(media_key, {})
                        if media_obj.get('url'):
                            media_urls.append(media_obj['url'])
            
            if media_urls:
                content['media'] = media_urls
            
            # 元数据（核心信息）- 版本化时区处理
            raw_timestamp = raw_message.get('timestamp', '')
            
            # 智能时区处理
            if isinstance(raw_timestamp, datetime):
                # datetime对象，转换为本地时区
                if raw_timestamp.tzinfo:
                    local_time = raw_timestamp.astimezone()
                else:
                    local_time = raw_timestamp
                posted_at = local_time.isoformat()
                timezone_info = str(local_time.tzinfo) if local_time.tzinfo else "本地时区"
            else:
                # 字符串格式，可能需要解析
                try:
                    dt = datetime.fromisoformat(str(raw_timestamp).replace('Z', '+00:00'))
                    if dt.tzinfo:
                        local_time = dt.astimezone()
                        posted_at = local_time.isoformat()
                        timezone_info = str(local_time.tzinfo)
                    else:
                        posted_at = str(raw_timestamp)
                        timezone_info = "inherited"
                except:
                    posted_at = str(raw_timestamp)
                    timezone_info = "inherited"
            
            metadata = {
                "posted_at": posted_at,
                "platform_specific": {
                    "message_id": message_id,
                    "channel_id": raw_message.get('channel_id', ''),
                    "guild_id": raw_message.get('guild_id', ''),
                    # 🎯 版本标识
                    "timezone_version": "v2.0_local",
                    "timezone_info": timezone_info
                }
            }
            
            # 处理reactions（如果有）
            reactions = {}
            discord_reactions = raw_message.get('reactions', [])
            for reaction in discord_reactions:
                if isinstance(reaction, dict):
                    emoji = reaction.get('emoji', {})
                    emoji_name = emoji.get('name', '') if emoji else ''
                    count = reaction.get('count', 0)
                    if emoji_name and count > 0:
                        reactions[emoji_name] = count
            
            if reactions:
                metadata['platform_specific']['reactions'] = reactions
            
            # 上下文（服务器/频道信息）
            context = {}
            
            channel_name = raw_message.get('channel_name', '')
            server_name = raw_message.get('guild_name', '')
            
            if channel_name:
                context['channel'] = channel_name
            if server_name:
                context['server'] = server_name
            
            result = {
                'id': unified_id,
                'platform': 'discord',
                'author': author,
                'content': content,
                'metadata': metadata,
                'context': context  # 总是包含context字段（可能为空），满足验证器要求
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"转换Discord消息时出错: {e}")
            return None
    
    def _is_actual_media(self, url: str) -> bool:
        """判断URL是否为实际媒体内容（非头像等）。"""
        if not url:
            return False
        
        # 排除头像URL
        avatar_indicators = [
            'profile_images',
            'profile_banners', 
            'avatars',
            'avatar',
            '_normal.jpg',
            '_mini.jpg'
        ]
        
        url_lower = url.lower()
        return not any(indicator in url_lower for indicator in avatar_indicators)
    
    def _validate_optimized_message(self, message: Dict[str, Any]) -> bool:
        """验证优化后的消息格式。"""
        required_fields = ['id', 'platform', 'author', 'content', 'metadata']
        
        for field in required_fields:
            if field not in message:
                self.logger.debug(f"验证失败：缺少字段 '{field}'")
                return False
        
        # 验证作者信息
        author = message.get('author', {})
        if not isinstance(author, dict):
            self.logger.debug(f"验证失败：author不是字典类型: {type(author)}")
            return False
        
        if not author.get('name'):
            self.logger.debug(f"验证失败：author.name为空: {author}")
            return False
            
        if not author.get('id'):
            self.logger.debug(f"验证失败：author.id为空: {author}")
            return False
        
        # 验证内容
        content = message.get('content', {})
        if not isinstance(content, dict):
            self.logger.debug(f"验证失败：content不是字典类型: {type(content)}")
            return False
        
        # 验证元数据
        metadata = message.get('metadata', {})
        if not isinstance(metadata, dict):
            self.logger.debug(f"验证失败：metadata不是字典类型: {type(metadata)}")
            return False
            
        if not metadata.get('posted_at'):
            self.logger.debug(f"验证失败：posted_at为空: {metadata}")
            return False
        
        return True
    
    def convert_messages_batch(self, raw_messages: List[Dict[str, Any]], 
                              platform: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        批量转换消息。
        
        Args:
            raw_messages: 原始消息列表
            platform: 平台名称
            
        Returns:
            (转换成功的消息列表, 错误信息列表)
        """
        converted_messages = []
        errors = []
        
        for i, raw_message in enumerate(raw_messages):
            try:
                converted = self.convert_message(raw_message, platform)
                if converted:
                    converted_messages.append(converted)
                else:
                    errors.append(f"消息 {i}: 转换失败")
            except Exception as e:
                errors.append(f"消息 {i}: {str(e)}")
        
        self.logger.info(
            f"批量转换完成: {len(converted_messages)}/{len(raw_messages)} 成功"
        )
        
        return converted_messages, errors