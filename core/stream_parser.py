# -*- coding: utf-8 -*-
"""流解析模块 - 处理 DeepSeek SSE 流响应"""
import json
import re

from .config import logger

# 预编译正则表达式
_TOOL_CALL_PATTERN = re.compile(r'\{\s*["\']tool_calls["\']\s*:\s*\[(.*?)\]\s*\}', re.DOTALL)
_CITATION_PATTERN = re.compile(r"^\[citation:")


def parse_deepseek_sse_line(raw_line: bytes) -> dict | None:
    """解析 DeepSeek SSE 行
    
    Args:
        raw_line: 原始字节行
        
    Returns:
        解析后的 chunk 字典，如果解析失败或应跳过则返回 None
    """
    try:
        line = raw_line.decode("utf-8")
    except Exception as e:
        logger.warning(f"[parse_deepseek_sse_line] 解码失败: {e}")
        return None
    
    if not line or not line.startswith("data:"):
        return None
    
    data_str = line[5:].strip()
    
    if data_str == "[DONE]":
        return {"type": "done"}
    
    try:
        chunk = json.loads(data_str)
        return chunk
    except json.JSONDecodeError as e:
        logger.warning(f"[parse_deepseek_sse_line] JSON解析失败: {e}")
        return None


def extract_content_from_chunk(chunk: dict) -> tuple[str, str, bool]:
    """从 DeepSeek chunk 中提取内容
    
    Args:
        chunk: 解析后的 chunk 字典
        
    Returns:
        (content, content_type, is_finished) 元组
        content_type 为 "thinking" 或 "text"
        is_finished 为 True 表示响应结束
    """
    if chunk.get("type") == "done":
        return "", "text", True
    
    # 检测内容审核/敏感词阻止
    if "error" in chunk or chunk.get("code") == "content_filter":
        logger.warning(f"[extract_content_from_chunk] 检测到内容过滤: {chunk}")
        return "", "text", True
    
    if "v" not in chunk:
        return "", "text", False
    
    v_value = chunk["v"]
    ptype = "text"
    
    # 检查路径确定类型
    path = chunk.get("p", "")
    if path == "response/search_status":
        return "", "text", False  # 跳过搜索状态
    elif path == "response/thinking_content":
        ptype = "thinking"
    elif path == "response/content":
        ptype = "text"
    
    if isinstance(v_value, str):
        if v_value == "FINISHED":
            return "", ptype, True
        return v_value, ptype, False
    elif isinstance(v_value, list):
        for item in v_value:
            if item.get("p") == "status" and item.get("v") == "FINISHED":
                return "", ptype, True
        return "", ptype, False
    
    return "", ptype, False


def collect_deepseek_response(response) -> tuple[str, str]:
    """收集 DeepSeek 流响应的完整内容
    
    Args:
        response: DeepSeek 流响应对象
        
    Returns:
        (reasoning_content, text_content) 元组
    """
    thinking_parts = []
    text_parts = []
    
    try:
        for raw_line in response.iter_lines():
            chunk = parse_deepseek_sse_line(raw_line)
            if not chunk:
                continue
            
            content, content_type, is_finished = extract_content_from_chunk(chunk)
            
            if is_finished:
                break
            
            if content:
                if content_type == "thinking":
                    thinking_parts.append(content)
                else:
                    text_parts.append(content)
    except Exception as e:
        logger.error(f"[collect_deepseek_response] 收集响应失败: {e}")
    finally:
        try:
            response.close()
        except Exception:
            pass
    
    return "".join(thinking_parts), "".join(text_parts)


def parse_tool_calls(text: str, tools_requested: list) -> list[dict]:
    """从响应文本中解析工具调用
    
    Args:
        text: 响应文本
        tools_requested: 请求中定义的工具列表
        
    Returns:
        检测到的工具调用列表，每项包含 name 和 input
    """
    detected_tools = []
    cleaned_text = text.strip()
    
    # 尝试直接解析完整 JSON
    if cleaned_text.startswith('{"tool_calls":') and cleaned_text.endswith("]}"):
        try:
            tool_data = json.loads(cleaned_text)
            for tool_call in tool_data.get("tool_calls", []):
                tool_name = tool_call.get("name")
                tool_input = tool_call.get("input", {})
                if any(tool.get("name") == tool_name for tool in tools_requested):
                    detected_tools.append({"name": tool_name, "input": tool_input})
            if detected_tools:
                return detected_tools
        except json.JSONDecodeError:
            pass
    
    # 使用正则匹配
    matches = _TOOL_CALL_PATTERN.findall(cleaned_text)
    for match in matches:
        try:
            tool_calls_json = f'{{"tool_calls": [{match}]}}'
            tool_data = json.loads(tool_calls_json)
            for tool_call in tool_data.get("tool_calls", []):
                tool_name = tool_call.get("name")
                tool_input = tool_call.get("input", {})
                if any(tool.get("name") == tool_name for tool in tools_requested):
                    detected_tools.append({"name": tool_name, "input": tool_input})
        except json.JSONDecodeError:
            continue
    
    return detected_tools


def should_filter_citation(text: str, search_enabled: bool) -> bool:
    """检查是否应该过滤引用内容
    
    Args:
        text: 内容文本
        search_enabled: 是否启用搜索
        
    Returns:
        是否应该过滤
    """
    if not search_enabled:
        return False
    return _CITATION_PATTERN.match(text) is not None
