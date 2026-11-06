# -*- coding: utf-8 -*-
"""
Day 2 · 手写工具调用循环（不用任何框架）

目标：让模型通过调用 calculator 工具，正确算出 987654 * 321098

核心认知：
    模型只会输出文本，不会执行任何东西。
    "调用工具" = 模型输出一段结构化文本 + 你的代码去执行 + 把结果发回去。

运行：
    python tool_calling.py
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────────
API_URL = "https://www.geek2api.com/v1/responses"
MODEL = "deepseek-v4.1-flash"
CRED_FILE = Path.home() / ".dsh" / ".credentials.yaml"


def load_api_key():
    """从 DSH 的凭据文件里读出 GEEK1_API_KEY"""
    text = CRED_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "GEEK1_API_KEY" in line:
            return line.split(":", 1)[1].strip()
    raise RuntimeError("没找到 GEEK1_API_KEY")


# ── 工具定义 ────────────────────────────────────────────────
# 注意：这里只是"告诉模型有这么个工具"，模型看不到函数实现
# TOOLS = [
#     {
#         "type": "function",
#         "name": "calculator",
#         "description": "计算数学表达式，支持 + - * / ** 和括号。用于精确的数值计算。",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "expression": {
#                     "type": "string",
#                     "description": "要计算的数学表达式，例如 '1234*5678'",
#                 }
#             },
#             "required": ["expression"],
#         },
#     }
# ]

TOOLS = [
    {
        "type": "function",
        "name": "translate",
        "description": "把文本翻译成指定语言。",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要翻译的文本"},
                "target_lang": {"type": "string", "description": "目标语言，如 en、zh"},
            },
            "required": ["text", "target_lang"],
        },
    }
]

def calculator(expression):
    """真正执行计算的函数（模型调不到它，是你的代码在调）

    注意：生产环境绝不能用 eval！这里只是教学演示。
    正确做法是用 ast.literal_eval 或专门的表达式解析库。
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return f"错误：表达式包含非法字符：{expression}"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算失败：{e}"


# 工具名 → 实际函数 的映射表
TOOL_IMPL = {
    "calculator": calculator,
}


# ── API 调用 ────────────────────────────────────────────────
def call_model(api_key, input_items, tools=None, retries=3):
    """发一次请求给模型，返回 (文本内容, 工具调用列表)

    返回的 tool_calls 形如：
        [{"name": "calculator", "arguments": {"expression": "1234*5678"}, "call_id": "..."}]

    网络抖动（DNS 解析失败、连接超时）会自动重试，间隔 2s、4s 递增。
    """
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

    # ── 带重试的请求（指数退避）────────────────────────────
    data = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            # 4xx 是客户端错误（参数/鉴权问题），重试没有意义，直接抛
            if 400 <= e.code < 500:
                raise RuntimeError(
                    f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}"
                )
            # 5xx 服务端错误可以重试
            if attempt == retries:
                raise RuntimeError(f"HTTP {e.code}，已重试 {retries} 次")
            wait = attempt * 2
            print(f"[重试 {attempt}/{retries}] HTTP {e.code}，{wait} 秒后重试...")
            time.sleep(wait)
        except (urllib.error.URLError, OSError) as e:
            # DNS 解析失败、连接超时、连接被重置等网络层问题
            if attempt == retries:
                raise RuntimeError(f"网络请求失败（已重试 {retries} 次）：{e}")
            wait = attempt * 2
            print(f"[重试 {attempt}/{retries}] 网络异常：{e}，{wait} 秒后重试...")
            time.sleep(wait)

    if data is None:
        raise RuntimeError("请求未获得任何响应")

    # ── 解析响应 ──────────────────────────────────────────
    # 这是今天的核心：从 output 里分别取出"文本"和"工具调用"
    text_parts = []
    tool_calls = []

    for item in data.get("output", []):
        item_type = item.get("type")

        if item_type == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))

        elif item_type == "function_call":
            # 模型说"我要调用某个工具"
            raw_args = item.get("arguments", "{}")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "name": item.get("name"),
                "arguments": args,
                "call_id": item.get("call_id") or item.get("id"),
            })

    return "".join(text_parts), tool_calls


# ── 主循环（今天你要写的核心）────────────────────────────────
def run_agent(api_key, question, max_rounds=5):
    """带工具调用的对话循环

    流程：
        1. 把问题发给模型
        2. 如果模型要调工具 → 执行工具 → 把结果拼进对话 → 回到 1
        3. 如果模型直接回答 → 结束
    """
    # 对话历史，每轮都完整发回去（模型没有记忆）
    input_items = [
        {"role": "user", "content": [{"type": "input_text", "text": question}]}
    ]

    for round_no in range(1, max_rounds + 1):
        print(f"\n--- 第 {round_no} 轮 ---")

        text, tool_calls = call_model(api_key, input_items, TOOLS)

        if text:
            print(f"[模型文本] {text[:100]}")

        # ── 情况 A：模型要调工具 ────────────────────────────
        if tool_calls:
            print(f"[模型请求] 调用 {len(tool_calls)} 个工具")

            for call in tool_calls:
                name = call["name"]
                args = call["arguments"]
                call_id = call["call_id"]
                print(f"  → {name}({args})")

                # TODO 1: 执行工具
                #   提示：从 TOOL_IMPL 里按 name 取出函数
                #   参数是 args["expression"]
                #   如果工具不存在，返回一个错误字符串
                # result = None  # ← 替换这行
                func = TOOL_IMPL.get(name)
                if func is None:
                    result = f"错误：没有名为 {name} 的工具"
                else:
                    #result = func(args["expression"])
                    result = func(args.get("expression", ""))

                print(f"  ← 结果: {result}")

                # TODO 2: 把"模型要调工具"这件事记进对话历史
                #   格式：
                #   {"type": "function_call", "name": name,
                #    "arguments": json.dumps(args), "call_id": call_id}
                # ← 你的代码
                input_items.append({
                    "type": "function_call",
                    "name": name,
                    "arguments": json.dumps(args),
                    "call_id": call_id,
                })

                # TODO 3: 把工具结果记进对话历史
                #   格式：
                #   {"type": "function_call_output", "call_id": call_id, "output": result}
                # ← 你的代码
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                })

                # 我错了，模型错了，只有代码对了。
                #
                # 这是今天最该记住的一课：
                #
                # 在这个问题上，人和模型都会“看起来很有把握地给出错误答案”。唯一可靠的是可验证的执行。
                #
                # 而
                # Agent
                # 的价值，就是把「不可靠的推测」换成「可靠的计算」。

            # 回到循环开头，把新历史再发给模型
            continue

        # ── 情况 B：模型直接回答，结束 ──────────────────────
        if text:
            return text

        # ── 情况 C：既没文本也没工具调用（异常情况）────────
        print("[警告] 模型既没回答也没调工具，结束")
        return None

    return f"[已达最大轮数 {max_rounds}，强制停止]"


def main():
    api_key = load_api_key()

    # 这道题模型单独算不准（我们实测过，它每次给的答案都不一样）
    question =  "请用 translate 工具把'你好'翻译成英文" #"987654 * 321098 等于多少？请用 calculator 工具计算，不要自己算。"

    print(f"问题：{question}")
    answer = run_agent(api_key, question)
    print(f"\n{'='*50}\n最终答案：{answer}")


if __name__ == "__main__":
    main()
