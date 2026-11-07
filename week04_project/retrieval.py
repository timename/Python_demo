# -*- coding: utf-8 -*-
"""
检索模块：BM25 关键词检索

从 week03 的 rag_length.py 抽取，改造成可复用的模块。
外部只需调用：index = ReportIndex(text)，然后 index.search(question)
"""

import math
import re
from collections import Counter


# ── 分词 ────────────────────────────────────────────────────
def tokenize(text):
    """把文本切成检索单元：中文按字切，英文/数字按词切"""
    text = text.lower()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


# ── 文档切分 ────────────────────────────────────────────────
def chunk_report(text):
    """按语义小节切分报告

    分节标记：
        - 一、二、三… 开头
        - 数字 + 点开头，且不以「；。，」结尾（排除列表项）
    跳过纯 "=" 分隔线。
    """
    chunks = []
    current_title = "报告开头"
    current_lines = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append({
                "id": len(chunks),
                "title": current_title,
                "content": body,
            })

    for line in text.splitlines():
        stripped = line.strip()

        if re.fullmatch(r"=+", stripped):
            continue

        is_big_section = re.match(r"^[一二三四五六七八九十]+、", stripped)
        is_sub_section = (
            re.match(r"^\d+\.\s*", stripped)
            and not stripped.endswith(("；", "。", ";", "，"))
        )

        if is_big_section or is_sub_section:
            flush()
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return chunks


# ── 索引 ────────────────────────────────────────────────────
class ReportIndex:
    """一份文档的检索索引（BM25）

    用法：
        idx = ReportIndex(text)
        results = idx.search("血糖怎么样")
    """

    def __init__(self, text, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunk_report(text)
        self._build()

    def _build(self):
        """建词频表和 IDF"""
        self.index = []
        for chunk in self.chunks:
            tokens = tokenize(chunk["title"] + " " + chunk["content"])
            self.index.append({
                **chunk,
                "tf": Counter(tokens),
                "length": len(tokens),
            })

        total = len(self.index)
        df = Counter()
        for item in self.index:
            for token in item["tf"]:
                df[token] += 1

        self.idf = {
            token: math.log((total + 1) / (freq + 1)) + 1
            for token, freq in df.items()
        }

        self.avg_length = sum(i["length"] for i in self.index) / max(len(self.index), 1)

    def _score(self, question_tokens, item):
        """BM25 打分"""
        length_penalty = (1 - self.b) + self.b * (item["length"] / self.avg_length)
        score = 0.0
        for token in question_tokens:
            tf = item["tf"].get(token, 0)
            if tf == 0:
                continue
            weight = self.idf.get(token, 0)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * length_penalty)
            score += tf_norm * weight
        return score

    def search(self, question, top_k=3, min_score=3.0):
        """检索最相关的 top_k 个片段

        返回：[{"id", "title", "content", "score"}, ...]
        """
        question_tokens = tokenize(question)
        scored = []

        for item in self.index:
            s = self._score(question_tokens, item)
            if s > min_score:
                scored.append({
                    "id": item["id"],
                    "title": item["title"],
                    "content": item["content"],
                    "score": round(s, 2),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_all_titles(self):
        """列出所有小节标题（调试用）"""
        return [c["title"] for c in self.chunks]
