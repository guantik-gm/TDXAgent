"""
Prompt template management system for TDXAgent.

This module provides a comprehensive system for managing, customizing,
and optimizing prompt templates for different platforms and use cases.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from string import Template
import re

from utils.logger import TDXLogger
from utils.helpers import ensure_directory, sanitize_filename


class PromptTemplate:
    """
    Individual prompt template with metadata and validation.
    """
    
    def __init__(self, name: str, template: str, 
                 platform: str = "", 
                 description: str = "",
                 variables: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize prompt template.
        
        Args:
            name: Template name
            template: Template string with placeholders
            platform: Target platform
            description: Template description
            variables: List of expected variables
            metadata: Additional metadata
        """
        self.name = name
        self.template = template
        self.platform = platform
        self.description = description
        self.variables = variables or self._extract_variables()
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # Validate template
        self._validate()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        # Find all {variable} patterns
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, self.template)
        return list(set(matches))
    
    def _validate(self) -> None:
        """Validate template format and variables."""
        if not self.template:
            raise ValueError("Template cannot be empty")
        
        # Check for balanced braces
        open_braces = self.template.count('{')
        close_braces = self.template.count('}')
        if open_braces != close_braces:
            raise ValueError("Unbalanced braces in template")
        
        # Validate variable names
        for var in self.variables:
            if not var.isidentifier():
                raise ValueError(f"Invalid variable name: {var}")
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.
        
        Args:
            **kwargs: Variables to substitute
            
        Returns:
            Formatted template string
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable: {missing_var}")
        except Exception as e:
            raise ValueError(f"Template formatting error: {e}")
    
    def validate_variables(self, variables: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate that all required variables are provided.
        
        Args:
            variables: Dictionary of variables
            
        Returns:
            Tuple of (is_valid, missing_variables)
        """
        missing = [var for var in self.variables if var not in variables]
        return len(missing) == 0, missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            'name': self.name,
            'template': self.template,
            'platform': self.platform,
            'description': self.description,
            'variables': self.variables,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create template from dictionary."""
        template = cls(
            name=data['name'],
            template=data['template'],
            platform=data.get('platform', ''),
            description=data.get('description', ''),
            variables=data.get('variables'),
            metadata=data.get('metadata')
        )
        
        # Restore timestamps if available
        if 'created_at' in data:
            template.created_at = data['created_at']
        if 'updated_at' in data:
            template.updated_at = data['updated_at']
        
        return template
    
    def update(self, template: Optional[str] = None,
               description: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update template properties."""
        if template is not None:
            self.template = template
            self.variables = self._extract_variables()
            self._validate()
        
        if description is not None:
            self.description = description
        
        if metadata is not None:
            self.metadata.update(metadata)
        
        self.updated_at = datetime.now().isoformat()


class PromptManager:
    """
    Comprehensive prompt template management system.
    
    Features:
    - Template storage and retrieval
    - Platform-specific templates
    - Template versioning
    - Custom template creation
    - Template validation and testing
    - Import/export functionality
    """
    
    def __init__(self, templates_directory: Union[str, Path] = "prompts"):
        """
        Initialize prompt manager.
        
        Args:
            templates_directory: Directory to store templates
        """
        self.templates_directory = Path(templates_directory)
        self.logger = TDXLogger.get_logger("tdxagent.processors.prompts")
        
        # Template storage
        self.templates: Dict[str, PromptTemplate] = {}
        self.platform_templates: Dict[str, List[str]] = {}
        
        # Default templates
        self.default_templates = self._get_default_templates()
        
        # Mark as not initialized - will be done on first use
        self._initialized = False
    
    async def _initialize(self) -> None:
        """Initialize the prompt manager."""
        ensure_directory(self.templates_directory)
        
        # Load existing templates
        await self.load_templates()
        
        # Create default templates if none exist
        if not self.templates:
            await self._create_default_templates()
        
        self._initialized = True
        self.logger.info(f"Initialized prompt manager with {len(self.templates)} templates")
    
    async def _ensure_initialized(self) -> None:
        """Ensure the prompt manager is initialized."""
        if not self._initialized:
            await self._initialize()
    
    def _get_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get default template definitions."""
        return {
            'twitter_summary': {
                'template': '''请分析以下 Twitter 内容，提取 3-5 个关键话题和重要信息。

注意：每条消息的时间后已包含可点击的作者链接，在分析时可直接引用。

内容：
{data}

请按以下格式输出：
## 关键话题
1. 话题1：简要描述 (来自[CryptoAnalyst的推文](https://x.com/user/status/123))
2. 话题2：简要描述 (来自[TechReporter的推文](https://x.com/user/status/456))

## 重要信息摘要
- 重要信息1 (来自[作者名的推文](链接))
- 重要信息2 (来自[作者名的推文](链接))

## 值得关注的内容
- 值得关注的推文或讨论 (来自[作者名的推文](链接))

引用说明：Telegram群组消息已按群组合并显示，分析时请使用简化引用格式，如 (来自 群组名 @用户名 时间)。''',
                'platform': 'twitter',
                'description': 'Twitter内容分析和总结模板',
                'metadata': {'type': 'summary', 'language': 'zh'}
            },
            
            'telegram_summary': {
                'template': '''请总结以下 Telegram 群聊的重要信息和讨论要点。

注意：每条消息的时间后已包含可点击的作者链接，在分析时可直接引用。

群聊内容：
{data}

请按以下格式输出：
## 主要讨论话题
1. 话题1：讨论要点 (来自 黄金矿工版群组 @BitHappy、@RuudDuur 04:19-04:20)
2. 话题2：讨论要点 (来自 技术讨论群组 @开发者 15:30)

## 重要信息
- 重要信息1 (来自 黄金矿工版群组 @BitHappy 06:58)
- 重要信息2 (来自 交易策略群组 @高志 06:32)

## 值得关注的消息
- 重要消息或公告 (来自 黄金矿工版群组 @jimecroke 07:04)

引用说明：Telegram群组消息已按群组合并显示，分析时请使用简化引用格式，如 (来自 群组名 @用户名 时间)。''',
                'platform': 'telegram',
                'description': 'Telegram群聊内容总结模板',
                'metadata': {'type': 'summary', 'language': 'zh'}
            },
            
            'discord_summary': {
                'template': '''请分析以下 Discord 讨论内容，提取核心要点和重要信息。

注意：每条消息的时间后已包含可点击的作者链接，在分析时可直接引用。

讨论内容：
{data}

请按以下格式输出：
## 讨论要点
1. 要点1：详细说明 (基于：消息1,3)
2. 要点2：详细说明 (基于：消息5,7)

## 重要决定或公告
- 重要决定1 (基于：消息2)
- 重要决定2 (基于：消息4,6)

## 技术讨论摘要
- 技术要点1 (基于：消息8)
- 技术要点2 (基于：消息9,10)

引用说明：在每个分析结论后用 (基于：消息1,2) 的格式标注支撑该结论的消息编号。''',
                'platform': 'discord',
                'description': 'Discord讨论内容分析模板',
                'metadata': {'type': 'summary', 'language': 'zh'}
            },
            
            'general_analysis': {
                'template': '''请分析以下社交媒体内容，提供详细的分析报告。

注意：每条消息的时间后已包含可点击的作者链接，在分析时可直接引用。

内容：
{data}

分析要求：
- 识别主要话题和趋势 (引用相关消息编号)
- 提取关键信息和观点 (引用相关消息编号)
- 分析情感倾向 (引用相关消息编号)
- 总结重要结论 (引用相关消息编号)

请提供结构化的分析报告，并在每个分析结论后使用 (基于：消息1,2) 格式标注支撑该结论的消息编号。''',
                'platform': '',
                'description': '通用社交媒体内容分析模板',
                'metadata': {'type': 'analysis', 'language': 'zh'}
            },
            
            # Integration templates for progressive analysis
            'twitter_integration': {
                'template': '''请将新的 Twitter 数据分析与之前的分析结果进行整合。

{integration_context}

新 Twitter 内容：
{data}

整合要求：
1. 将新数据的分析与前批次结果整合，形成统一连贯的完整分析
2. 避免重复已分析的内容，重点关注新的话题和趋势
3. 保持所有引用链接的完整性
4. 确保最终结果具有整体性和逻辑连贯性
5. 如发现新的重要话题或与前批次相关的内容，请特别突出

请按以下格式输出整合后的完整分析：
## 🔥 热门话题
1. 话题描述 (来自[作者名的推文](链接))

## 💡 重要信息摘要
- 重要信息 (来自[作者名的推文](链接))

## 📈 趋势观察
- 趋势分析 (来自[作者名的推文](链接))

**重要引用要求**：
- 必须使用消息引用表中提供的完整链接
- 禁止使用 (消息1, 消息2) 这种格式
- 每个引用必须是可点击的Markdown链接
- 示例：正确格式 → (来自[作者名的推文](https://x.com/xxx/status/xxx))
- 示例：错误格式 → (消息1, 20, 153) ❌''',
                'platform': 'twitter',
                'description': 'Twitter渐进式整合分析模板',
                'metadata': {'type': 'integration', 'language': 'zh'}
            },
            
            'telegram_integration': {
                'template': '''请将新的 Telegram 群聊内容与之前的分析结果进行整合。

{integration_context}

新 Telegram 群聊内容：
{data}

整合要求：
1. 将新群聊内容的分析与前批次结果整合，形成统一连贯的完整分析
2. 避免重复已分析的讨论要点，重点关注新的话题和重要信息
3. 保持消息引用的完整性
4. 确保最终结果反映群聊的整体讨论脉络
5. 如发现与前批次相关或延续的讨论，请特别说明

请按以下格式输出整合后的完整分析：
## 🗣️ 主要讨论话题
1. 话题描述 (基于：消息引用表中的详细描述)

## 📢 重要信息
- 重要信息 (基于：消息引用表中的详细描述)

## 🔍 值得关注的内容
- 重要消息或公告 (基于：消息引用表中的详细描述)

**重要引用要求**：
- 必须使用消息引用表中提供的完整消息描述
- 禁止使用 ([消息1], [消息2]) 这种格式
- Telegram消息使用详细描述: (来自某群组 @用户名 时间的消息)
- 示例：正确格式 → (来自Gate.io官方群 @管理员 08-01 10:30的消息)
- 示例：错误格式 → ([消息6], [消息9]) ❌''',
                'platform': 'telegram',
                'description': 'Telegram渐进式整合分析模板',
                'metadata': {'type': 'integration', 'language': 'zh'}
            },
            
            'discord_integration': {
                'template': '''请将新的 Discord 讨论内容与之前的分析结果进行整合。

{integration_context}

新 Discord 讨论内容：
{data}

整合要求：
1. 将新讨论内容的分析与前批次结果整合，形成统一连贯的完整分析
2. 避免重复已分析的讨论要点，重点关注新的决定、技术讨论或重要更新
3. 保持消息引用的完整性
4. 确保最终结果反映社区讨论的整体发展
5. 如发现与前批次相关或延续的技术讨论，请特别说明

请按以下格式输出整合后的完整分析：
## 💬 讨论要点
1. 要点描述 (基于：消息1,3)

## 📋 重要决定或公告
- 重要决定 (基于：消息2)

## 🔧 技术讨论摘要
- 技术要点 (基于：消息8)

引用说明：请使用消息编号格式进行引用，如 (基于：消息1,2)。''',
                'platform': 'discord',
                'description': 'Discord渐进式整合分析模板',
                'metadata': {'type': 'integration', 'language': 'zh'}
            },
            
            'general_integration': {
                'template': '''请将新的社交媒体内容与之前的分析结果进行整合。

{integration_context}

新内容：
{data}

整合要求：
1. 将新数据的分析与前批次结果整合，形成统一连贯的完整分析报告
2. 避免重复已分析的内容，重点关注新的话题、趋势和重要信息
3. 保持所有引用的完整性和准确性
4. 确保最终结果具有整体性和逻辑连贯性
5. 识别跨批次的关联和发展趋势

分析要求：
- 整合主要话题和趋势分析 (引用相关消息)
- 综合重要信息和观点 (引用相关消息)
- 统一情感倾向分析 (引用相关消息)
- 形成完整的结论总结 (引用相关消息)

请提供结构化的整合分析报告，并在每个分析结论后标注支撑该结论的消息引用。''',
                'platform': '',
                'description': '通用渐进式整合分析模板',
                'metadata': {'type': 'integration', 'language': 'zh'}
            }
        }
    
    async def _create_default_templates(self) -> None:
        """Create default templates."""
        for name, template_data in self.default_templates.items():
            template = PromptTemplate(
                name=name,
                template=template_data['template'],
                platform=template_data['platform'],
                description=template_data['description'],
                metadata=template_data['metadata']
            )
            
            await self.add_template(template)
        
        self.logger.info("Created default templates")
    
    async def add_template(self, template: PromptTemplate) -> None:
        """
        Add a new template.
        
        Args:
            template: PromptTemplate instance
        """
        # Validate template
        if template.name in self.templates:
            raise ValueError(f"Template '{template.name}' already exists")
        
        # Add to storage
        self.templates[template.name] = template
        
        # Update platform index
        if template.platform:
            if template.platform not in self.platform_templates:
                self.platform_templates[template.platform] = []
            self.platform_templates[template.platform].append(template.name)
        
        # Save to file
        await self._save_template(template)
        
        self.logger.info(f"Added template: {template.name}")
    
    async def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate instance or None
        """
        await self._ensure_initialized()
        return self.templates.get(name)
    
    async def get_platform_templates(self, platform: str) -> List[PromptTemplate]:
        """
        Get all templates for a platform.
        
        Args:
            platform: Platform name
            
        Returns:
            List of PromptTemplate instances
        """
        template_names = self.platform_templates.get(platform, [])
        return [self.templates[name] for name in template_names if name in self.templates]
    
    async def update_template(self, name: str, 
                            template: Optional[str] = None,
                            description: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing template.
        
        Args:
            name: Template name
            template: New template string
            description: New description
            metadata: New metadata
            
        Returns:
            True if updated successfully
        """
        if name not in self.templates:
            return False
        
        # Update template
        self.templates[name].update(template, description, metadata)
        
        # Save changes
        await self._save_template(self.templates[name])
        
        self.logger.info(f"Updated template: {name}")
        return True
    
    async def delete_template(self, name: str) -> bool:
        """
        Delete a template.
        
        Args:
            name: Template name
            
        Returns:
            True if deleted successfully
        """
        if name not in self.templates:
            return False
        
        template = self.templates[name]
        
        # Remove from platform index
        if template.platform and template.platform in self.platform_templates:
            if name in self.platform_templates[template.platform]:
                self.platform_templates[template.platform].remove(name)
        
        # Remove from storage
        del self.templates[name]
        
        # Delete file
        template_file = self.templates_directory / f"{sanitize_filename(name)}.yaml"
        if template_file.exists():
            template_file.unlink()
        
        self.logger.info(f"Deleted template: {name}")
        return True
    
    async def list_templates(self, platform: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            platform: Filter by platform (optional)
            
        Returns:
            List of template names
        """
        if platform:
            return self.platform_templates.get(platform, [])
        else:
            return list(self.templates.keys())
    
    async def format_template(self, name: str, **kwargs) -> str:
        """
        Format a template with variables.
        
        Args:
            name: Template name
            **kwargs: Variables for formatting
            
        Returns:
            Formatted template string
        """
        template = await self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        return template.format(**kwargs)
    
    async def validate_template_variables(self, name: str, 
                                        variables: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate variables for a template.
        
        Args:
            name: Template name
            variables: Variables to validate
            
        Returns:
            Tuple of (is_valid, missing_variables)
        """
        template = await self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        return template.validate_variables(variables)
    
    async def load_templates(self) -> None:
        """Load templates from files."""
        if not self.templates_directory.exists():
            return
        
        for template_file in self.templates_directory.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                template = PromptTemplate.from_dict(data)
                self.templates[template.name] = template
                
                # Update platform index
                if template.platform:
                    if template.platform not in self.platform_templates:
                        self.platform_templates[template.platform] = []
                    if template.name not in self.platform_templates[template.platform]:
                        self.platform_templates[template.platform].append(template.name)
                
            except Exception as e:
                self.logger.error(f"Failed to load template from {template_file}: {e}")
    
    async def _save_template(self, template: PromptTemplate) -> None:
        """Save template to file."""
        template_file = self.templates_directory / f"{sanitize_filename(template.name)}.yaml"
        
        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    async def export_templates(self, export_path: Union[str, Path], 
                             platform: Optional[str] = None) -> bool:
        """
        Export templates to a file.
        
        Args:
            export_path: Path to export file
            platform: Filter by platform (optional)
            
        Returns:
            True if exported successfully
        """
        try:
            templates_to_export = {}
            
            for name, template in self.templates.items():
                if platform is None or template.platform == platform:
                    templates_to_export[name] = template.to_dict()
            
            export_file = Path(export_path)
            
            if export_file.suffix.lower() == '.json':
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(templates_to_export, f, ensure_ascii=False, indent=2)
            else:
                with open(export_file, 'w', encoding='utf-8') as f:
                    yaml.dump(templates_to_export, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Exported {len(templates_to_export)} templates to {export_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export templates: {e}")
            return False
    
    async def import_templates(self, import_path: Union[str, Path], 
                             overwrite: bool = False) -> int:
        """
        Import templates from a file.
        
        Args:
            import_path: Path to import file
            overwrite: Whether to overwrite existing templates
            
        Returns:
            Number of templates imported
        """
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                raise FileNotFoundError(f"Import file not found: {import_file}")
            
            # Load data
            with open(import_file, 'r', encoding='utf-8') as f:
                if import_file.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            imported_count = 0
            
            for name, template_data in data.items():
                if name in self.templates and not overwrite:
                    self.logger.warning(f"Skipping existing template: {name}")
                    continue
                
                try:
                    template = PromptTemplate.from_dict(template_data)
                    
                    # Remove existing template if overwriting
                    if name in self.templates:
                        await self.delete_template(name)
                    
                    await self.add_template(template)
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to import template '{name}': {e}")
            
            self.logger.info(f"Imported {imported_count} templates from {import_file}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Failed to import templates: {e}")
            return 0
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about templates."""
        stats = {
            'total_templates': len(self.templates),
            'platforms': {},
            'template_types': {},
            'languages': {}
        }
        
        for template in self.templates.values():
            # Platform stats
            platform = template.platform or 'general'
            stats['platforms'][platform] = stats['platforms'].get(platform, 0) + 1
            
            # Type stats
            template_type = template.metadata.get('type', 'unknown')
            stats['template_types'][template_type] = stats['template_types'].get(template_type, 0) + 1
            
            # Language stats
            language = template.metadata.get('language', 'unknown')
            stats['languages'][language] = stats['languages'].get(language, 0) + 1
        
        return stats
