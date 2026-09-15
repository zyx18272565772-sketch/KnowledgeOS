"""用户认证 — 注册 / 登录"""
import hashlib
import os
import secrets
import logging
import time
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from core.mysql_client import mysql_client

logger = logging.getLogger(__name__)
router = APIRouter()
TOKEN_TTL_SECONDS = 8 * 60 * 60
_sessions: dict[str, dict] = {}
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin").strip()
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "").strip()


def _create_session(username: str, is_admin: bool) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "username": username,
        "is_admin": is_admin,
        "expires_at": time.time() + TOKEN_TTL_SECONDS,
    }
    return token


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """验证 Bearer Token，返回当前登录用户。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")

    token = authorization.removeprefix("Bearer ").strip()
    session = _sessions.get(token)
    if not session or session["expires_at"] <= time.time():
        _sessions.pop(token, None)
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return session


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """限制仅管理员访问的后台接口。"""
    if not current_user["is_admin"]:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user


def _hash_password(password: str, salt: str = "") -> tuple[str, str]:
    """SHA-256 哈希密码，返回 (hash, salt)"""
    if not salt:
        salt = secrets.token_hex(16)
    h = hashlib.sha256((password + salt).encode()).hexdigest()
    return h, salt


def _ensure_table():
    """确保 users 表存在，并按环境变量创建初始管理员。"""
    mysql_client._ensure_connected()
    conn = mysql_client.connection
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                salt VARCHAR(64) NOT NULL,
                is_admin TINYINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE username = %s", (DEFAULT_ADMIN_USERNAME,))
        if cur.fetchone():
            return

    if not DEFAULT_ADMIN_PASSWORD:
        logger.warning(
            "尚无初始管理员且未配置 DEFAULT_ADMIN_PASSWORD，已跳过创建"
        )
        return

    with conn.cursor() as cur:
        h, s = _hash_password(DEFAULT_ADMIN_PASSWORD)
        cur.execute(
            "INSERT INTO users (username, password_hash, salt, is_admin) VALUES (%s, %s, %s, %s)",
            (DEFAULT_ADMIN_USERNAME, h, s, 1)
        )
    conn.commit()
    logger.info("已根据环境变量创建初始管理员：%s", DEFAULT_ADMIN_USERNAME)


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/register")
async def register(req: RegisterRequest):
    """注册新用户"""
    username = req.username.strip()
    password = req.password.strip()

    if len(username) < 2:
        raise HTTPException(400, "用户名至少 2 个字符")
    if len(password) < 3:
        raise HTTPException(400, "密码至少 3 个字符")

    _ensure_table()
    mysql_client._ensure_connected()
    conn = mysql_client.connection

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            raise HTTPException(409, "用户名已存在")

    h, s = _hash_password(password)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (%s, %s, %s)",
            (username, h, s)
        )
    conn.commit()
    logger.info(f"User registered: {username}")
    return {"ok": True, "username": username, "is_admin": False}


@router.post("/auth/login")
async def login(req: LoginRequest):
    """登录"""
    _ensure_table()
    mysql_client._ensure_connected()
    conn = mysql_client.connection

    with conn.cursor() as cur:
        cur.execute(
            "SELECT password_hash, salt, is_admin FROM users WHERE username = %s",
            (req.username.strip(),)
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(401, "用户名或密码错误")

    stored_hash, salt, is_admin = row
    h, _ = _hash_password(req.password.strip(), salt)

    if h != stored_hash:
        raise HTTPException(401, "用户名或密码错误")

    logger.info(f"User logged in: {req.username} (admin={is_admin})")
    token = _create_session(req.username, bool(is_admin))
    return {
        "ok": True,
        "username": req.username,
        "is_admin": bool(is_admin),
        "access_token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_TTL_SECONDS,
    }


@router.get("/users")
async def list_users(_current_user: dict = Depends(require_admin)):
    """用户列表（管理后台用）"""
    _ensure_table()
    mysql_client._ensure_connected()
    conn = mysql_client.connection
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, username, is_admin, created_at FROM users ORDER BY id ASC"
        )
        rows = cur.fetchall()
    users = [
        {
            "id": r[0],
            "username": r[1],
            "is_admin": bool(r[2]),
            "created_at": r[3].isoformat() if r[3] else "",
        }
        for r in rows
    ]
    return {"users": users, "total": len(users)}
