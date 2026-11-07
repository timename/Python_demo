# -*- coding: utf-8 -*-
"""模块自测：验证 retrieval 和 tools 能独立工作（不调模型）"""

from pathlib import Path

from retrieval import ReportIndex
from tools import build_tool_registry

text = Path("data/report.txt").read_text(encoding="utf-8")
idx = ReportIndex(text)

print("=== 切分结果 ===")
for c in idx.chunks:
    print(f"  [{c['id']}] {c['title']}  ({len(c['content'])} 字)")

reg = build_tool_registry(idx, text)

print("\n=== 工具自测 ===")

print("\n[search_report('血糖')]")
print(reg["search_report"]("血糖")[:200])

print("\n[get_indicator('血糖')]")
print(reg["get_indicator"]("血糖"))

print("\n[get_indicator('ALT')]")
print(reg["get_indicator"]("ALT"))

print("\n[get_indicator('血型')]  ← 报告中不存在")
print(reg["get_indicator"]("血型"))

print("\n[list_abnormal()]")
print(reg["list_abnormal"]())

print("\n[calculate_bmi(172, 78)]")
print(reg["calculate_bmi"](172, 78))

print("\n[calculate_bmi('abc', 78)]  ← 错误参数")
print(reg["calculate_bmi"]("abc", 78))

print("\n[calculate_bmi(0, 78)]  ← 非法值")
print(reg["calculate_bmi"](0, 78))
