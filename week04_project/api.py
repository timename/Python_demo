# -*- coding: utf-8 -*-
"""
体检报告智能助手 · FastAPI Web 服务

把 week04 的命令行 Agent 改造成 HTTP 服务。

启动：
    uv run uvicorn api:app --reload --port 8000

访问：
    http://127.0.0.1:8000/health          健康检查
    http://127.0.0.1:8000/docs            自动生成的 API 文档（Swagger）
    http://127.0.0.1:8000/ask?q=血糖多少   提问

Day 1 目标：跑通两个接口，理解 FastAPI 的三件套（app / route / 参数）
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from agent import load_api_key, run_agent
from retrieval import ReportIndex
from tools import build_tool_registry

BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "data" / "report.txt"


# ══════════════════════════════════════════════════════════
# 全局状态：启动时加载一次，所有请求共享
# ══════════════════════════════════════════════════════════
class AppState:
    api_key: str = ""
    index: ReportIndex = None
    registry: dict = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时执行一次，关闭时执行一次

    为什么放这里：索引构建要读文件、分词、算 IDF，耗时几百毫秒。
    每个请求都重建太浪费，所以启动时建一次，全局共享。
    """
    print("正在加载体检报告索引...")
    text = REPORT_FILE.read_text(encoding="utf-8")
    state.index = ReportIndex(text)
    state.registry = build_tool_registry(state.index, text)
    state.api_key = load_api_key()
    print(f"索引就绪：{len(state.index.chunks)} 个片段")
    yield
    print("服务关闭")


app = FastAPI(
    title="体检报告智能助手",
    description="基于 RAG + Agent 的体检报告问答服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════
# 数据模型（Pydantic）
# ══════════════════════════════════════════════════════════
class AskRequest(BaseModel):
    """POST /ask 的请求体

    Pydantic 的作用：自动校验 + 自动生成 API 文档
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="要问的问题，如「我的血糖是多少」",
        examples=["我的血糖是多少？"],
    )
    save_trace: bool = Field(
        default=False,
        description="是否保存 trace 到文件",
    )


class AskResponse(BaseModel):
    """POST /ask 的响应体"""
    question: str = Field(description="原始问题")
    answer: str = Field(description="模型回答")
    steps: int = Field(description="执行步数")
    tools_used: list[str] = Field(default_factory=list, description="用到的工具")
    trace: list[dict] = Field(default_factory=list, description="完整执行轨迹")


# ══════════════════════════════════════════════════════════
# 路由 1：健康检查
# ══════════════════════════════════════════════════════════
@app.get("/health", summary="健康检查")
def health():
    """最简单的接口，用于确认服务活着

    返回服务状态和已加载的资源信息。
    """
    return {
        "status": "ok",
        "chunks": len(state.index.chunks) if state.index else 0,
        "tools": list(state.registry.keys()) if state.registry else [],
    }


# ══════════════════════════════════════════════════════════
# 路由 0：根路径，重定向到文档页
# ══════════════════════════════════════════════════════════
@app.get("/", include_in_schema=False)
def root():
    """访问根路径时自动跳到 API 文档页

    include_in_schema=False 表示这个路由不出现在 /docs 的接口列表里，
    因为它只是个跳转，不算业务接口。
    """
    return RedirectResponse(url="/docs")


# ══════════════════════════════════════════════════════════
# 路由 2：GET 提问（简单，参数在 URL 里）
# ══════════════════════════════════════════════════════════
@app.get("/ask", summary="提问（GET，参数在 URL）")
def ask_get(
    q: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="要问的问题",
        examples=["我的血糖是多少？"],
    )
):
    """GET 方式的提问，方便浏览器直接测试

    注意：GET 有 URL 长度限制，长问题请用 POST /ask。
    """
    return _do_ask(q)


# ══════════════════════════════════════════════════════════
# 路由 3：POST 提问（正式接口，请求体传参）
# ══════════════════════════════════════════════════════════
@app.post("/ask", response_model=AskResponse, summary="提问（POST，请求体）")
def ask_post(req: AskRequest):
    """POST 方式的提问，支持更长的文本和结构化响应

    请求示例：
        POST /ask
        {"question": "我的血糖是多少？"}
    """
    return _do_ask(req.question)


# ══════════════════════════════════════════════════════════
# 内部实现
# ══════════════════════════════════════════════════════════
def _do_ask(question: str) -> dict:
    """两个路由共用的执行逻辑

    TODO 1: 调用 run_agent 并组装响应
        1. 检查 state.registry 是否就绪，没有就抛 HTTPException(503)
        2. 调用 run_agent(state.api_key, question, state.registry, verbose=False)
           返回 (answer, trace)
        3. 从 trace 里提取用到的工具名：
           [step["tool"] for step in trace if step.get("type") == "tool_call"]
        4. 返回字典：
           {"question": question, "answer": answer,
            "steps": len(trace), "tools_used": [...], "trace": trace}
        5. 用 try/except 包住，出错抛 HTTPException(500, detail=str(e))
    """
    # TODO 1: 你的代码

    # 1. 检查就绪
    if state.registry is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    # 2. 调用 Agent
    try:
        answer, trace = run_agent(state.api_key, question, state.registry, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. 提取工具名
    tools_used = [step["tool"] for step in trace if step.get("type") == "tool_call"]

    # 4. 返回
    return {
        "question": question,
        "answer": answer,
        "steps": len(trace),
        "tools_used": tools_used,
        "trace": trace,
    }




if __name__ == "__main__":
    import uvicorn

    # 在 PyCharm / VS Code 里直接点运行即可启动。
    # 注意：这里传的是 app 对象而不是 "api:app" 字符串，
    # 因为字符串形式依赖工作目录，在 IDE 里容易报 Could not import module "api"。
    # 代价是不能用 reload=True（reload 要求传字符串）。
    # 需要热重载时，用命令行：
    #     uv run uvicorn api:app --reload --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
