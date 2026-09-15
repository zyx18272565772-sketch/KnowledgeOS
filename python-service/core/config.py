import os
import logging
from typing import Dict, Any

class ConfigManager:
    """统一配置管理模块"""
    
    def __init__(self):
        """初始化配置管理器"""
        self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """加载配置项"""
        # AI Embeddings
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
        self.LLM_REQUEST_TIMEOUT_SECONDS = float(
            os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "30")
        )
        self.LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
        # Embedding模型选择: "dashscope" 使用云端API, "local" 使用本地中文模型
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "dashscope")
        # 本地Embedding模型名称（仅当EMBEDDING_MODEL=local时生效）
        self.LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        
        # Milvus Configuration
        self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
        self.MILVUS_USER = os.getenv("MILVUS_USER", "")
        self.MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")

        # Redis Configuration
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        # 当前会话的附件只缓存 OCR 文本，不保存原文件。默认 60 分钟后过期。
        self.ATTACHMENT_CACHE_TTL = int(os.getenv("ATTACHMENT_CACHE_TTL", "3600"))

        # MySQL Configuration
        self.DB_HOST = os.getenv("MYSQL_HOST", "localhost")
        self.DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
        self.DB_USER = os.getenv("MYSQL_USERNAME", "root")
        self.DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
        self.DB_NAME = os.getenv("MYSQL_DATABASE", "ai_knowledge_db")
        
        # Vector Store Configuration
        self.USE_MILVUS = os.getenv("USE_MILVUS", "false").lower() == "true"
        self.VECTOR_STORE_PERSIST_DIR = os.getenv("VECTOR_STORE_PERSIST_DIR", "./faiss_index")
        self.VECTOR_STORE_COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION_NAME", "ai_knowledge_collection")
        # 待审核文档的向量缓存。缓存与线上 FAISS 完全分离，只有审核通过后
        # 才会写入正式索引。
        self.PENDING_VECTOR_DIR = os.getenv("PENDING_VECTOR_DIR", "./pending_vectors")
        
        # Rerank Configuration
        self.RERANKER_TYPE = os.getenv("RERANKER_TYPE", "simple")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

        # Retrieval Configuration
        self.KNOWLEDGE_TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "5"))
        self.REASONING_TOP_K = int(os.getenv("REASONING_TOP_K", "3"))
        self.RETRIEVAL_CANDIDATE_MULTIPLIER = int(
            os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "3")
        )

        # Knowledge duplicate inspection. Semantic inspection reuses vectors
        # already stored in FAISS and never re-embeds the knowledge base.
        self.DUPLICATE_TOP_K = int(os.getenv("DUPLICATE_TOP_K", "5"))
        self.DUPLICATE_COSINE_THRESHOLD = float(
            os.getenv("DUPLICATE_COSINE_THRESHOLD", "0.90")
        )
        self.DUPLICATE_MAX_CANDIDATES = int(
            os.getenv("DUPLICATE_MAX_CANDIDATES", "20")
        )
        self.DUPLICATE_LLM_VERIFY = os.getenv(
            "DUPLICATE_LLM_VERIFY", "true"
        ).lower() in {"true", "1", "yes", "on"}
        # 新文档与已发布知识做预发布巡检时，阈值略低于“全库查重”，
        # 让关键数字不同的潜在规则冲突也有机会进入 LLM Judge。
        self.INSPECTION_COSINE_THRESHOLD = float(
            os.getenv("INSPECTION_COSINE_THRESHOLD", "0.80")
        )
        
        # Text Chunking Configuration
        self.CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "semantic")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))
        
        # Tesseract OCR Configuration（默认为空，由 parser.py 自动检测）
        self.TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
        
        # Temporary Files Configuration
        self.TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
        # 确保临时目录存在
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        
        # CORS Configuration
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        # 第三方库会输出大量英文初始化细节；项目保留 WARNING/ERROR，隐藏 INFO 噪声。
        for logger_name in (
            "faiss.loader",
            "httpx",
            "httpx2",
            "mysql.connector",
            "sentence_transformers",
            "huggingface_hub",
        ):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        self.logger.info("系统配置加载完成")
    
# 创建全局配置实例
config = ConfigManager()
