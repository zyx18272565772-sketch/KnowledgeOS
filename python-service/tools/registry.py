from typing import Dict, Any, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from tools.base import Tool
from core.config import config


class ToolRegistry:
    """工具注册器"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tools = {}
        return cls._instance
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self._tools[tool.name] = tool
        config.logger.info(f"工具注册完成：{tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools
    
    def invoke_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # 验证输入参数
        if not tool.validate_input(parameters):
            raise ValueError(f"Invalid input parameters for tool: {tool_name}")
        
        # 执行工具（带超时和重试）
        retries = 0
        max_retries = tool.metadata.max_retries
        timeout_ms = tool.metadata.timeout_ms
        
        timeout_sec = timeout_ms / 1000.0

        while retries <= max_retries:
            start_time = time.time()
            try:
                #使用线程池执行工具，强制超时控制
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(tool.execute, parameters)
                    try:
                        result = future.result(timeout=timeout_sec)
                    except FutureTimeoutError:
                        future.cancel()
                        raise TimeoutError(f"Tool {tool_name} timed out after {timeout_ms}ms")

                execution_time = (time.time() - start_time) * 1000
                config.logger.info(f"工具 {tool_name} 执行完成，耗时 {execution_time:.2f} 毫秒")

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                retries += 1
                if retries > max_retries:
                    config.logger.error(f"工具 {tool_name} 重试 {max_retries} 次后仍失败：{e}")
                    raise
                config.logger.warning(f"工具 {tool_name} 第 {retries}/{max_retries} 次执行失败：{e}")
                time.sleep(0.5)
    
# 全局工具注册器实例
tool_registry = ToolRegistry()
