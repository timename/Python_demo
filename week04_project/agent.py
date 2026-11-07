# -*- coding: utf-8 -*-
"""
体检报告智能助手 · Agent 主程序

架构：
    用户提问
       ↓
    Agent 循环（模型决定调哪个工具）
       ├─ search_report  RAG 检索
       ├─ get_indicator  精确取值
       ├─ list_abnormal  列异常项
       └─ calculate_bmi  算 BMI
       ↓
    最终答案 + trace 日志（可回放、可调试）

运行：
    python agent.py                      # 跑全部测试用例
    python agent.py --question "血糖怎么样"
    python agent.py --trace              # 额外保存 trace 到 traces/
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from retrieval import ReportIndex
from tools import TOOL_SCHEMAS, build_tool_registry

API_URL = "https://www.geek2api.com/v1/responses"
MODEL = "deepseek-v4.1-flash"
CRED_FILE = Path.home() / ".dsh" / ".credentials.yaml"
BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "data" / "report.txt"
TRACE_DIR = BASE_DIR / "traces"

MAX_ROUNDS = 6

SYSTEM_PROMPT = """你是一个体检报告解读助手。你可以调用工具查询报告内容。

工作原则：
1. 必须用工具获取数据，不要凭记忆或常识作答。
2. 已知具体指标名时，用 get_indicator 精确查询（比检索准确）。
3. 需要上下文、建议、整体情况时，用 search_report 检索。
4. 用户问「有哪些问题」时，用 list_abnormal。
5. 工具返回「未找到」「未提及」时，如实告诉用户报告中未提及，不要猜测。
6. 回答要标注数据来源，格式 [来源：小节名]。
7. 用中文回答，简洁直接。"""


def load_api_key():
    text = CRED_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "GEEK1_API_KEY" in line:
            return line.split(":", 1)[1].strip()
    raise RuntimeError("没找到 GEEK1_API_KEY")


def call_model(api_key, input_items, tools=None, retries=3):
    """调用模型，返回 (文本, 工具调用列表)"""
    body = {
        "model": MODEL,
        "input": input_items,
        "max_output_tokens": 1024,
        "temperature": 0,
    }
    if tools:
        body["tools"] = tools

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    data = None
    for attempt in range(1, retries + 1):
        try:
            # 只对这个请求禁用系统代理。
            # 原因：Windows 上开着 Clash 等代理软件时，urllib 会自动走系统代理，
            # 而代理在转发到本 API 时会中断连接，表现为
            #   SSL: UNEXPECTED_EOF_WHILE_READING（固定 5 秒后失败）
            # 手工 socket / http.client / node fetch 都不读系统代理，所以不受影响。
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}")
            if attempt == retries:
                raise RuntimeError(f"HTTP {e.code}，已重试 {retries} 次")
            time.sleep(attempt * 2)
        except (urllib.error.URLError, OSError) as e:
            if attempt == retries:
                raise RuntimeError(f"网络失败（已重试 {retries} 次）：{e}")
            print(f"[重试 {attempt}/{retries}] {e}")
            time.sleep(attempt * 2)

    if data is None:
        raise RuntimeError("未获得响应")

    text_parts = []
    tool_calls = []
    for item in data.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))
        elif item_type == "function_call":
            raw = item.get("arguments", "{}")
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "name": item.get("name"),
                "arguments": args,
                "call_id": item.get("call_id") or item.get("id"),
            })

    return "".join(text_parts), tool_calls


# ══════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════
def run_agent(api_key, question, registry, verbose=True):
    """Agent 主循环，返回 (最终答案, trace)

    trace 是一个列表，记录每一步发生了什么：
        {"round": 1, "type": "tool_call", "tool": "get_indicator",
         "args": {...}, "result": "...", "elapsed_ms": 12}
        {"round": 1, "type": "text", "text": "..."}
    """
    trace = []
    input_items = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "input_text", "text": question}]},
    ]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"问题：{question}")

    for round_no in range(1, MAX_ROUNDS + 1):
        text, tool_calls = call_model(api_key, input_items, TOOL_SCHEMAS)

        # 模型输出文本
        if text:
            if verbose:
                print(f"  [第{round_no}轮·模型] {text[:80]}...")
                # TODO 1: 把这条文本记录进 trace
                #   格式：{"round": round_no, "type": "text", "text": text}
                trace.append({"round": round_no, "type": "text", "text": text}) #出问题时能回看「模型当时说了什么」。

        # 没有工具调用 → 结束
        if not tool_calls:
            if text:
                return text, trace
            trace.append({"round": round_no, "type": "error", "message": "模型无输出"})
            return "（模型未返回内容）", trace

        # 执行工具
        for call in tool_calls:
            name = call["name"]
            args = call["arguments"]
            call_id = call["call_id"]

            if verbose:
                args_str = json.dumps(args, ensure_ascii=False) #中文必须设 False，否则 trace 文件全是 Unicode 转义，没法看。
                print(f"  [第{round_no}轮·调用] {name}({args_str[:60]})")

            # 执行
            start = time.time()
            func = registry.get(name)
            if func is None:
                result = f"错误：没有名为 {name} 的工具"
            else:
                try:
                    # TODO 2: 调用工具函数，参数来自 args
                    #   提示：args 是字典，要按参数名展开
                    #   search_report/get_indicator 只有一个参数 query/name
                    #   calculate_bmi 有两个参数 height_cm / weight_kg
                    #   用 func(**args) 可以自动按名字匹配
                    result = func(**args) #None          # ← 替换
                except Exception as e:
                    result = f"错误：工具执行失败 - {type(e).__name__}: {e}"
            elapsed_ms = int((time.time() - start) * 1000)

            if verbose:
                preview = str(result).replace("\n", " ")[:70]
                print(f"  [第{round_no}轮·结果] {preview}  ({elapsed_ms}ms)")

            # TODO 3: 把这次工具调用记进 trace
            #   格式：
            #   {"round": round_no, "type": "tool_call", "tool": name,
            #    "args": args, "result": str(result), "elapsed_ms": elapsed_ms}
            trace.append({
                "round": round_no,
                "type": "tool_call",
                "tool": name,
                "args": args,
                "result": str(result),
                "elapsed_ms": elapsed_ms,
            })

            # 把调用和结果写进对话历史
            input_items.append({
                "type": "function_call",
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
                "call_id": call_id,
            })
            input_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": str(result),
            })

    # 超过最大轮数
    trace.append({"round": MAX_ROUNDS, "type": "error", "message": "达到最大轮数"})
    return f"（已达到最大轮数 {MAX_ROUNDS}，未能得出结论）", trace


# ══════════════════════════════════════════════════════════
# trace 保存与回放
# ══════════════════════════════════════════════════════════
def save_trace(question, answer, trace, path=None):
    """把一次完整对话保存成 JSON，便于事后排查

    TODO 4: 实现这个函数
        1. 组装 dict：
           {"question": question, "answer": answer,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "steps": len(trace), "trace": trace}
        2. 如果 path 为 None，自动生成文件名：
           TRACE_DIR / f"trace_{时间戳}.json"
           时间戳用 datetime.now().strftime("%Y%m%d_%H%M%S")
        3. TRACE_DIR.mkdir(parents=True, exist_ok=True)
        4. 写入 JSON，ensure_ascii=False, indent=2
        5. 返回写入的文件路径
    """
    # TODO 4: 你的代码
    record = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "steps": len(trace),
        "trace": trace,
    }

    if path is None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = TRACE_DIR / f"trace_{stamp}.json"

    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None)
    parser.add_argument("--trace", action="store_true", help="保存 trace 到文件")
    args = parser.parse_args()

    api_key = load_api_key()
    report_text = REPORT_FILE.read_text(encoding="utf-8")
    index = ReportIndex(report_text)
    registry = build_tool_registry(index, report_text)

    print(f"索引就绪：{len(index.chunks)} 个片段")
    print(f"可用工具：{', '.join(registry.keys())}")

    if args.question:
        answer, trace = run_agent(api_key, args.question, registry)
        print(f"\n【答案】\n{answer}")
        print(f"\n【trace】{len(trace)} 步")
        for step in trace:
            print(f"  round{step['round']} {step['type']}: "
                  f"{step.get('tool', '')} {str(step.get('result', ''))[:50]}")
        if args.trace:
            p = save_trace(args.question, answer, trace)
            print(f"\n已保存 trace: {p}")
        return

    # 测试用例
    cases = [
        # (问题, 期望行为)
        ("我的血糖是多少？", "get_indicator 或 search_report，答出 7.2"),
        ("肝功能有问题吗？", "答出 ALT 55 偏高"),
        ("报告里有哪些异常指标？", "list_abnormal，列出 5 条"),
        ("我身高172，体重78，帮我算下BMI", "calculate_bmi"),
        ("报告里的医生是谁？", "search_report，答王医生"),
        ("我的血型是什么？", "任何工具都查不到 → 应回答未提及"),
    ]

    result_summary = []
    for i, (q, expect) in enumerate(cases, 1):
        print(f"\n\n{'#' * 60}")
        print(f"# 用例 {i}/{len(cases)}  期望：{expect}")
        try:
            answer, trace = run_agent(api_key, q, registry)
            print(f"\n【答案】{answer}")
            tools_used = [s.get("tool") for s in trace if s["type"] == "tool_call"]
            print(f"【用到工具】{tools_used or '（无）'}  共 {len(trace)} 步")
            if args.trace:
                save_trace(q, answer, trace)
            result_summary.append({"q": q, "ok": True, "answer_len": len(answer)})
        except Exception as e:
            print(f"\n【失败】{type(e).__name__}: {e}")
            result_summary.append({"q": q, "ok": False, "error": str(e)})

    print(f"\n\n{'=' * 60}")
    print(f"全部完成：{sum(1 for r in result_summary if r['ok'])}/{len(cases)} 成功")


if __name__ == "__main__":
    main()
