"""测试 RouterAgent 路由Agent"""

import json
from unittest.mock import Mock

import pytest
from workflows.router_agent import RouterAgent, TaskType
from intent.classifier import IntentResult, IntentType


@pytest.fixture
def router():
    return RouterAgent()


class TestClassifyTask:
    """测试任务分类"""

    def test_classify_chitchat_greeting(self, router):
        """测试识别闲聊-问候"""
        assert router.classify_task("你好") == TaskType.CHITCHAT

    def test_classify_chitchat_farewell(self, router):
        """测试识别闲聊-告别"""
        assert router.classify_task("谢谢，再见") == TaskType.CHITCHAT

    def test_classify_chitchat_identity(self, router):
        """测试识别闲聊-身份询问"""
        assert router.classify_task("你是谁") == TaskType.CHITCHAT

    def test_classify_chitchat_daily(self, router):
        """测试识别闲聊-日常"""
        assert router.classify_task("讲个笑话") == TaskType.CHITCHAT

    def test_classify_chitchat_weather(self, router):
        """测试识别闲聊-天气"""
        assert router.classify_task("今天天气怎么样") == TaskType.CHITCHAT

    def test_classify_knowledge_qa_technical(self, router):
        """测试识别知识问答-技术"""
        assert router.classify_task("什么是 RAG？请解释它的工作原理") == TaskType.KNOWLEDGE_QA

    def test_classify_knowledge_qa_programming(self, router):
        """测试识别知识问答-编程"""
        assert router.classify_task("Python 的装饰器怎么用？") == TaskType.KNOWLEDGE_QA

    def test_classify_knowledge_qa_database(self, router):
        """测试识别知识问答-数据库"""
        assert router.classify_task("MySQL 的事务隔离级别有哪些？") == TaskType.KNOWLEDGE_QA

    def test_classify_knowledge_qa_concept(self, router):
        """测试识别知识问答-概念（含'区别'且>15字，走推理链路）"""
        assert router.classify_task("深度学习和机器学习的区别是什么？") == TaskType.REASONING

    def test_classify_simple_complexity(self, router):
        assert router.classify_complexity("报销规则是什么") == "simple"

    def test_classify_short_comparison_as_complex(self, router):
        assert router.classify_complexity("对比方案A和方案B他俩的优缺点") == "complex"

    def test_classify_contextual_complexity(self, router):
        assert router.classify_complexity("这个规则怎么理解", has_context=True) == "medium"

    def test_new_conversation_does_not_imply_context(self, router):
        assert router.classify_complexity("报销规则是什么", has_context=False) == "simple"

    def test_classify_inspection(self, router):
        """聊天入口不再触发知识巡检 Agent。"""
        result = router.classify_task("检查重复文档")
        assert result != TaskType.KNOWLEDGE_INSPECTION

    def test_classify_inspection_low_quality(self, router):
        """低质量巡检也应从知识库管理页触发。"""
        result = router.classify_task("检查低质量片段")
        assert result != TaskType.KNOWLEDGE_INSPECTION

    def test_classify_unknown_short_text(self, router):
        """测试短文本无关键词时归类为闲聊"""
        result = router.classify_task("帮我写一篇文章")
        # 短文本（<10字符）且无疑问词，默认闲聊
        assert result == TaskType.CHITCHAT

    def test_classify_unknown_long_text_defaults_to_knowledge(self, router):
        """测试长文本含'分析'时走推理链路"""
        result = router.classify_task("请帮我分析一下这个问题的具体情况并给出建议")
        assert result == TaskType.REASONING


class TestParseInspectionType:
    """测试巡检类型解析"""

    def test_parse_duplicate(self, router):
        """测试解析重复类型"""
        assert router._parse_inspection_type("检查重复文档") == "duplicate"

    def test_parse_low_quality(self, router):
        """测试解析低质量类型"""
        assert router._parse_inspection_type("检查低质量内容") == "low_quality"

    def test_parse_stale(self, router):
        """测试解析过期类型"""
        assert router._parse_inspection_type("检查过期文档") == "stale"

    def test_parse_full(self, router):
        """测试默认全量巡检"""
        assert router._parse_inspection_type("知识库巡检") == "full"


class TestInspectionChatRemoval:
    """即使旧意图模型返回 inspection，也会安全回落到知识问答。"""

    def test_legacy_inspection_intent_maps_to_knowledge_qa(self, router):
        router.classifier.classify = Mock(return_value=IntentResult(
            intent=IntentType.KNOWLEDGE_INSPECTION,
            reasoning="旧模型输出",
        ))
        assert router.classify_task("巡检知识库") == TaskType.KNOWLEDGE_QA


class TestGetTaskStats:
    """测试任务统计"""

    def test_stats_has_all_keys(self, router):
        """测试统计包含所有键"""
        stats = router.get_task_stats()
        assert "chitchat_keywords" in stats
        assert "knowledge_qa_keywords" in stats
        assert "emotion_keywords" in stats

    def test_stats_values_positive(self, router):
        """测试统计值为正数"""
        stats = router.get_task_stats()
        for count in stats.values():
            assert count > 0
