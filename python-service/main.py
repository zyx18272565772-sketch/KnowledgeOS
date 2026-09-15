from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path
import os
from dotenv import load_dotenv

# 加载环境变量 (确保加载当前目录下的 .env 文件)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from api.agent_routes import router as agent_router
from api.document_routes import router as document_router
from api.auth import router as auth_router
import tools

BASE_DIR = Path(__file__).parent.parent  # D:\Agent\my agent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="KnowledgeOS - 企业智能知识平台")

# 配置 CORS
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(agent_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(auth_router, prefix="/api")

# 前端样式与脚本。业务接口保持原路径不变。
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 前端静态页面
@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

# 健康检查端点
@app.get("/health")
async def health_check():
    # 检查环境变量
    has_api_key = os.getenv("DASHSCOPE_API_KEY") is not None
    use_milvus = os.getenv("USE_MILVUS", "false").lower() == "true"
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")

    # 检查向量存储
    vector_store_info = {}
    if use_milvus:
        vector_store_info = {
            "type": "Milvus",
            "host": milvus_host,
            "port": milvus_port,
            "status": "configured"
        }
    else:
        vector_store_dir = os.path.join(os.getcwd(), "faiss_index")
        vector_store_exists = os.path.exists(vector_store_dir)
        vector_store_info = {
            "type": "FAISS",
            "exists": vector_store_exists,
            "directory": vector_store_dir
        }

    return {
        "status": "healthy",
        "environment": {
            "has_dashscope_api_key": has_api_key,
            "use_milvus": use_milvus
        },
        "vector_store": vector_store_info
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
