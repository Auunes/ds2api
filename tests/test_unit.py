#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS2API 单元测试

测试核心模块的功能，不依赖网络请求
"""
import json
import os
import sys
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig(unittest.TestCase):
    """配置模块测试"""

    def test_config_loading(self):
        """测试配置加载"""
        from core.config import load_config, CONFIG
        
        # 测试加载函数不会抛出异常
        config = load_config()
        self.assertIsInstance(config, dict)

    def test_config_paths(self):
        """测试配置路径"""
        from core.config import WASM_PATH, CONFIG_PATH
        
        # 路径应该是字符串
        self.assertIsInstance(WASM_PATH, str)
        self.assertIsInstance(CONFIG_PATH, str)


class TestMessages(unittest.TestCase):
    """消息处理模块测试"""

    def test_messages_prepare_simple(self):
        """测试简单消息处理"""
        from core.messages import messages_prepare
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        result = messages_prepare(messages)
        self.assertIn("Hello", result)

    def test_messages_prepare_multi_turn(self):
        """测试多轮对话消息处理"""
        from core.messages import messages_prepare
        
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = messages_prepare(messages)
        
        # 检查助手消息标签
        self.assertIn("<｜Assistant｜>", result)
        self.assertIn("<｜end▁of▁sentence｜>", result)
        # 检查用户消息标签
        self.assertIn("<｜User｜>", result)

    def test_messages_prepare_array_content(self):
        """测试数组格式内容处理"""
        from core.messages import messages_prepare
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                    {"type": "image", "url": "http://example.com/image.png"}
                ]
            }
        ]
        result = messages_prepare(messages)
        
        self.assertIn("First part", result)
        self.assertIn("Second part", result)

    def test_markdown_image_removal(self):
        """测试 markdown 图片格式移除"""
        from core.messages import messages_prepare
        
        messages = [
            {"role": "user", "content": "Check this ![alt](http://example.com/image.png) image"}
        ]
        result = messages_prepare(messages)
        
        # 图片格式应该被改为链接格式
        self.assertNotIn("![alt]", result)
        self.assertIn("[alt]", result)

    def test_merge_consecutive_messages(self):
        """测试连续相同角色消息合并"""
        from core.messages import messages_prepare
        
        messages = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
            {"role": "user", "content": "Part 3"}
        ]
        result = messages_prepare(messages)
        
        self.assertIn("Part 1", result)
        self.assertIn("Part 2", result)
        self.assertIn("Part 3", result)

    def test_convert_claude_to_deepseek(self):
        """测试 Claude 到 DeepSeek 格式转换"""
        from core.messages import convert_claude_to_deepseek
        
        claude_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are helpful.",
            "temperature": 0.7,
            "stream": True
        }
        
        result = convert_claude_to_deepseek(claude_request)
        
        # 检查模型映射
        self.assertIn("deepseek", result.get("model", "").lower())
        
        # 检查 system 消息插入
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "You are helpful.")
        
        # 检查其他参数
        self.assertEqual(result.get("temperature"), 0.7)
        self.assertEqual(result.get("stream"), True)


class TestPow(unittest.TestCase):
    """PoW 模块测试"""

    def test_wasm_caching(self):
        """测试 WASM 缓存功能"""
        from core.pow import _get_cached_wasm_module, _wasm_module, _wasm_engine
        from core.config import WASM_PATH
        
        # 首次调用
        engine1, module1 = _get_cached_wasm_module(WASM_PATH)
        self.assertIsNotNone(engine1)
        self.assertIsNotNone(module1)
        
        # 再次调用应该返回相同的实例
        engine2, module2 = _get_cached_wasm_module(WASM_PATH)
        self.assertIs(engine1, engine2)
        self.assertIs(module1, module2)

    def test_get_account_identifier(self):
        """测试账号标识获取"""
        from core.utils import get_account_identifier
        
        # 测试邮箱
        account1 = {"email": "test@example.com"}
        self.assertEqual(get_account_identifier(account1), "test@example.com")
        
        # 测试手机号
        account2 = {"mobile": "13800138000"}
        self.assertEqual(get_account_identifier(account2), "13800138000")
        
        # 邮箱优先
        account3 = {"email": "test@example.com", "mobile": "13800138000"}
        self.assertEqual(get_account_identifier(account3), "test@example.com")


class TestSessionManager(unittest.TestCase):
    """会话管理器模块测试"""

    def test_get_model_config(self):
        """测试模型配置获取"""
        from core.session_manager import get_model_config
        
        # deepseek-chat
        thinking, search = get_model_config("deepseek-chat")
        self.assertEqual(thinking, False)
        self.assertEqual(search, False)
        
        # deepseek-reasoner
        thinking, search = get_model_config("deepseek-reasoner")
        self.assertEqual(thinking, True)
        self.assertEqual(search, False)
        
        # deepseek-chat-search
        thinking, search = get_model_config("deepseek-chat-search")
        self.assertEqual(thinking, False)
        self.assertEqual(search, True)
        
        # deepseek-reasoner-search
        thinking, search = get_model_config("deepseek-reasoner-search")
        self.assertEqual(thinking, True)
        self.assertEqual(search, True)
        
        # 大小写不敏感
        thinking, search = get_model_config("DeepSeek-CHAT")
        self.assertEqual(thinking, False)
        self.assertEqual(search, False)
        
        # 无效模型
        thinking, search = get_model_config("invalid-model")
        self.assertIsNone(thinking)
        self.assertIsNone(search)


class TestAuth(unittest.TestCase):
    """认证模块测试"""

    def test_auth_key_check(self):
        """测试 API Key 检查"""
        from core.config import CONFIG
        
        # 检查配置中是否有 keys
        keys = CONFIG.get("keys", [])
        self.assertIsInstance(keys, list)


class TestRegexPatterns(unittest.TestCase):
    """正则表达式测试"""

    def test_markdown_image_pattern(self):
        """测试 markdown 图片正则"""
        from core.messages import _MARKDOWN_IMAGE_PATTERN
        
        text = "Check ![alt text](http://example.com/image.png) here"
        match = _MARKDOWN_IMAGE_PATTERN.search(text)
        
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "alt text")
        self.assertEqual(match.group(2), "http://example.com/image.png")


if __name__ == "__main__":
    # 设置环境变量避免配置警告
    os.environ.setdefault("DS2API_CONFIG_PATH", 
                          os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"))
    
    unittest.main(verbosity=2)
