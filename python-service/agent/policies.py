from typing import Optional


class GuardrailsPolicy:
    """安全守卫策略"""

    def __init__(self):
        self.dangerous_keywords = [
            "hack", "exploit", "病毒", "木马",
            "攻击", "入侵", "破解", "作弊"
        ]

        self.sensitive_topics = [
            "政治", "宗教", "色情", "暴力",
            "赌博", "毒品", "犯罪"
        ]

    def check_input(self, text: str) -> tuple[bool, Optional[str]]:
        """检查输入是否安全"""
        if not text:
            return True, None

        lower_text = text.lower()

        for keyword in self.dangerous_keywords:
            if keyword in lower_text:
                return False, f"输入包含敏感关键词: {keyword}"

        for topic in self.sensitive_topics:
            if topic in text:
                return False, f"输入涉及敏感话题: {topic}"

        return True, None

guardrails = GuardrailsPolicy()
