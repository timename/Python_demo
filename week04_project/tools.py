# -*- coding: utf-8 -*-
"""
工具集：Agent 可以调用的函数

每个工具都是纯函数：输入参数 → 返回字符串（结果会回传给模型）
工具失败时返回以 "错误：" 开头的字符串，不抛异常（异常会让 Agent 崩掉）
"""

import re


# ── 工具 1：RAG 检索 ────────────────────────────────────────
def make_search_tool(index):
    """构造 search_report 工具（需要绑定索引，所以用工厂函数）"""

    def search_report(query):
        """在体检报告里检索相关段落

        Args:
            query: 检索关键词或问题，如 "血糖"、"肝功能"
        """
        results = index.search(query, top_k=3, min_score=3.0)
        if not results:
            return f"未找到与「{query}」相关的内容。"
        parts = []
        for r in results:
            parts.append(f"【{r['title']}】\n{r['content']}")
        return "\n\n---\n\n".join(parts)

    return search_report


# ── 工具 2：精确取值 ────────────────────────────────────────
# 报告里常见指标的别名映射
INDICATOR_ALIASES = {
    "血糖": ["血糖", "GLU", "葡萄糖"],
    "肝功能": ["谷丙转氨酶", "谷草转氨酶", "ALT", "AST"],
    "血脂": ["总胆固醇", "甘油三酯", "TC", "TG"],
    "血压": ["收缩压", "舒张压", "血压"],
    "BMI": ["BMI"],
    "体重": ["体重"],
    "身高": ["身高"],
    "心率": ["心率"],
    "血常规": ["白细胞", "红细胞", "血红蛋白"],
}


def make_indicator_tool(report_text):
    """构造 get_indicator 工具：从报告原文里精确抓取某个指标的行"""

    def get_indicator(name):
        """查询某个具体指标的数值

        Args:
            name: 指标名，如 "血糖"、"ALT"、"BMI"
        返回：
            匹配到的原文行（可能多行），找不到则返回提示
        """
        # 展开别名
        keywords = INDICATOR_ALIASES.get(name, [name])

        matched = []
        for line in report_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for kw in keywords:
                if kw.lower() in stripped.lower():
                    matched.append(stripped)
                    break

        if not matched:
            return f"报告中未找到指标「{name}」。"

        # 去重但保持顺序
        seen = set()
        unique = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                unique.append(m)

        return "\n".join(unique)

    return get_indicator


# ── 工具 3：列出异常项 ──────────────────────────────────────
def make_abnormal_tool(report_text):
    """构造 list_abnormal 工具（需要绑定报告原文）"""

    def list_abnormal():
        """列出报告里所有标记为异常的指标"""
        abnormal_lines = []
        for line in report_text.splitlines():
            stripped = line.strip()
            if re.search(r"\[(偏高|偏低|异常)", stripped):
                abnormal_lines.append(stripped)

        if not abnormal_lines:
            return "未发现标记为异常的指标。"

        return "\n".join(abnormal_lines)

    return list_abnormal


# ── 工具 4：计算 BMI ───────────────────────────────────────
def calculate_bmi(height_cm, weight_kg):
    """根据身高(cm)和体重(kg)计算 BMI

    Args:
        height_cm: 身高，单位厘米，如 172
        weight_kg: 体重，单位千克，如 78
    """
    try:
        h = float(height_cm)
        w = float(weight_kg)
    except (TypeError, ValueError):
        return f"错误：身高和体重必须是数字，收到 height={height_cm}, weight={weight_kg}"

    if h <= 0 or w <= 0:
        return f"错误：身高和体重必须大于 0，收到 height={h}, weight={w}"

    h_m = h / 100
    bmi = w / (h_m ** 2)

    if bmi < 18.5:
        level = "偏瘦"
    elif bmi < 24:
        level = "正常"
    elif bmi < 28:
        level = "超重"
    else:
        level = "肥胖"

    return f"BMI = {bmi:.1f}，属于「{level}」范围。（计算式：{w} / ({h_m:.2f}²)）"


# ── 工具定义（给模型看的 schema）────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "search_report",
        "description": "在体检报告中检索相关段落。用于查找某类信息、了解整体情况、或需要上下文时。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词，如「血糖」「肝功能」「建议」",
                }
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_indicator",
        "description": "精确查询某个具体指标的数值和参考范围。已知指标名称时优先用这个，比检索更准确。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "指标名，如「血糖」「ALT」「BMI」「血压」",
                }
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "list_abnormal",
        "description": "列出报告中所有异常指标。用户问「有哪些问题」「哪里不正常」时使用。",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "calculate_bmi",
        "description": "根据身高体重计算 BMI 并给出范围判断。用户想自己算 BMI 时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "height_cm": {"type": "number", "description": "身高，单位厘米"},
                "weight_kg": {"type": "number", "description": "体重，单位千克"},
            },
            "required": ["height_cm", "weight_kg"],
        },
    },
]


def build_tool_registry(index, report_text):
    """把工具名映射到实际函数

    这是 Agent 查表用的注册表。
    """
    return {
        "search_report": make_search_tool(index),
        "get_indicator": make_indicator_tool(report_text),
        "list_abnormal": make_abnormal_tool(report_text),
        "calculate_bmi": calculate_bmi,
    }
