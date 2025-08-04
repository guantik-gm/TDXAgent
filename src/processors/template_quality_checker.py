"""
模板质量检测器

基于提示词模板的 output_format 部分检测 AI 响应质量。
只要 AI 响应包含模板中定义的任何一个章节标题，就认为是有效响应。
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class QualityCheckResult:
    """质量检查结果"""
    is_valid: bool
    reason: str = ""
    found_sections: List[str] = None
    expected_sections: List[str] = None
    
    def __post_init__(self):
        if self.found_sections is None:
            self.found_sections = []
        if self.expected_sections is None:
            self.expected_sections = []


class TemplateQualityChecker:
    """基于模板的质量检测器"""
    
    def __init__(self):
        self.logger = logging.getLogger("tdxagent.template_quality_checker")
        self._cached_sections: Dict[str, List[str]] = {}
        self._template_cache: Dict[str, str] = {}
        
    def _load_template(self, template_name: str) -> Optional[str]:
        """加载模板内容，带缓存"""
        if template_name in self._template_cache:
            return self._template_cache[template_name]
            
        # 查找模板文件
        template_paths = [
            Path(f"prompts/{template_name}.yaml"),
            Path(f"src/prompts/{template_name}.yaml"),
            Path(f"../prompts/{template_name}.yaml"),
        ]
        
        for template_path in template_paths:
            if template_path.exists():
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self._template_cache[template_name] = content
                    return content
                except Exception as e:
                    self.logger.error(f"加载模板文件 {template_path} 失败: {e}")
                    continue
                    
        self.logger.error(f"未找到模板文件: {template_name}")
        return None
    
    def _extract_output_format_sections(self, template_content: str) -> List[str]:
        """从模板的 output_format 部分提取章节标题"""
        try:
            # 首先尝试解析为 YAML
            template_data = yaml.safe_load(template_content)
            if isinstance(template_data, dict) and 'template' in template_data:
                template_text = template_data['template']
            else:
                template_text = template_content
        except yaml.YAMLError:
            # 如果不是有效的 YAML，直接使用原文本
            template_text = template_content
        
        # 提取 <output_format> 到 </output_format> 之间的内容
        output_format_match = re.search(
            r'<output_format>(.*?)</output_format>', 
            template_text, 
            re.DOTALL | re.IGNORECASE
        )
        
        if not output_format_match:
            self.logger.warning("模板中未找到 <output_format> 部分")
            return []
            
        output_format_text = output_format_match.group(1)
        
        # 提取所有的章节标题（## 开头的行）
        section_pattern = r'^##\s+(.+)$'
        sections = []
        
        for line in output_format_text.split('\n'):
            line = line.strip()
            if match := re.match(section_pattern, line):
                section_title = match.group(1).strip()
                sections.append(section_title)
                
        self.logger.debug(f"从模板提取到 {len(sections)} 个章节标题: {sections}")
        return sections
    
    def _get_expected_sections(self, template_name: str) -> List[str]:
        """获取模板期望的章节标题，带缓存"""
        if template_name in self._cached_sections:
            return self._cached_sections[template_name]
            
        template_content = self._load_template(template_name)
        if not template_content:
            return []
            
        sections = self._extract_output_format_sections(template_content)
        self._cached_sections[template_name] = sections
        return sections
    
    def _normalize_section_title(self, title: str) -> str:
        """标准化章节标题，去除emoji和特殊字符，便于匹配"""
        import unicodedata
        import re
        
        # 去除emoji和符号字符
        normalized = ''.join(c for c in title if unicodedata.category(c) not in ['So', 'Sm', 'Sk'])
        # 去除多余的标点符号，但保留中文和英文字符
        normalized = re.sub(r'[^\w\u4e00-\u9fff\s]', '', normalized)
        # 去除多余空格
        normalized = ' '.join(normalized.split())
        return normalized.strip().lower()
    
    def _find_matching_sections(self, content: str, expected_sections: List[str]) -> List[str]:
        """在内容中查找匹配的章节标题"""
        found_sections = []
        content_lower = content.lower()
        
        for expected_section in expected_sections:
            # 完整匹配
            if expected_section in content:
                found_sections.append(expected_section)
                continue
                
            # 标准化后的匹配（去除emoji等）
            normalized_expected = self._normalize_section_title(expected_section)
            if normalized_expected and len(normalized_expected.strip()) > 2:
                if normalized_expected in content_lower:
                    found_sections.append(expected_section)
                    continue
                    
            # 关键词匹配（提取主要词汇）
            # 例如："💡 重要信息汇总" -> ["重要信息", "汇总"]
            keywords = [word for word in normalized_expected.split() 
                       if len(word) > 1 and word.isalpha()]
            if len(keywords) >= 2:
                if all(keyword in content_lower for keyword in keywords[:2]):
                    found_sections.append(expected_section)
                    
        return found_sections
    
    def is_valid_response(self, content: str, prompt_manager=None, 
                         template_name: str = 'pure_investment_analysis',
                         min_sections_required: int = 1) -> QualityCheckResult:
        """
        检查 AI 响应是否有效
        
        Args:
            content: AI 响应内容
            prompt_manager: 提示词管理器（兼容性参数，可为None）
            template_name: 模板名称
            min_sections_required: 最少需要的章节数量
            
        Returns:
            QualityCheckResult: 检查结果
        """
        if not content or not content.strip():
            return QualityCheckResult(
                is_valid=False,
                reason="内容为空",
                expected_sections=self._get_expected_sections(template_name)
            )
        
        expected_sections = self._get_expected_sections(template_name)
        if not expected_sections:
            # 如果无法获取期望章节，使用简单的章节标题检查
            has_any_section = bool(re.search(r'^##\s+', content, re.MULTILINE))
            return QualityCheckResult(
                is_valid=has_any_section,
                reason="无法获取模板期望章节，使用简单检查" if has_any_section else "未包含任何章节标题"
            )
        
        found_sections = self._find_matching_sections(content, expected_sections)
        
        is_valid = len(found_sections) >= min_sections_required
        
        if is_valid:
            reason = f"找到 {len(found_sections)} 个有效章节: {', '.join(found_sections[:3])}"
            if len(found_sections) > 3:
                reason += f" 等共{len(found_sections)}个"
        else:
            reason = f"未找到足够的有效章节（需要至少{min_sections_required}个，实际找到{len(found_sections)}个）"
        
        return QualityCheckResult(
            is_valid=is_valid,
            reason=reason,
            found_sections=found_sections,
            expected_sections=expected_sections
        )
    
    def clear_cache(self):
        """清除缓存"""
        self._cached_sections.clear()
        self._template_cache.clear()


# 全局实例
_quality_checker = None

def get_quality_checker() -> TemplateQualityChecker:
    """获取质量检测器实例（单例模式）"""
    global _quality_checker
    if _quality_checker is None:
        _quality_checker = TemplateQualityChecker()
    return _quality_checker