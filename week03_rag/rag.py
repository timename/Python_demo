# -*- coding: utf-8 -*-
"""
Day 3 · RAG 全链路（不用向量库，手写关键词检索）

目标：让模型基于体检报告原文回答问题，而不是靠记忆瞎编。

RAG 五步：
    ① 文档切分  chunk_report()
    ② 索引构建  build_index()
    ③ 检索      retrieve()      ← 今天你要写的核心
    ④ 拼接      build_prompt()
    ⑤ 生成      call_model()

运行：
    python rag.py
    python rag.py --question "我的血糖怎么样？"
"""

import argparse
import json
import math
import re
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────────
API_URL = "https://www.geek2api.com/v1/responses"
MODEL = "deepseek-v4.1-flash"
CRED_FILE = Path.home() / ".dsh" / ".credentials.yaml"
BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "data" / "report.txt"

TOP_K = 3          # 检索返回几个 chunk
MIN_SCORE = 3.0    # 分数低于此值视为不相关


def load_api_key():
    text = CRED_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "GEEK1_API_KEY" in line:
            return line.split(":", 1)[1].strip()
    raise RuntimeError("没找到 GEEK1_API_KEY")


# ══════════════════════════════════════════════════════════
# ① 文档切分
# ══════════════════════════════════════════════════════════
def chunk_report(text):
    """把报告按语义小节切成 chunk 列表

    策略：按行扫描，遇到「分节标记」就开新块。
    分节标记包括：
        - 以「一、」「二、」「三、」「四、」开头的行
        - 以数字+. 开头的行（如 "1. 血常规"、"2. 肝功能"）
        - 连续的 "=" 分隔线

    返回：[{"id": 0, "title": "个人信息", "content": "..."}, ...]

    提示：
        1. 用 text.splitlines() 按行遍历
        2. 维护 current_title / current_lines 两个变量
        3. 遇到分节标记 → 把上一块存起来，开新块
        4. 最后别忘了把最后一块也存进去
    """
    chunks = []
    current_title = "报告开头"
    current_lines = []

    # TODO 1: 实现切分逻辑
    #   判断一行是不是"分节标记"，可以用正则：
    #       re.match(r'^[一二三四五六七八九十]+、', line)      → 一、基础检查
    #       re.match(r'^\d+\.\s', line)                        → 1. 血常规
    #       re.match(r'^=+$', line.strip())                    → ==== 分隔线
    #
    #   遇到标记时：先把 current_lines 组装成 chunk 存入 chunks，
    #              再重置 current_lines 并更新 current_title
    def flush():
        """把当前累积的内容存入 chunks"""
        body = "\n".join(current_lines).strip()
        if body:  # 有内容才存
            chunks.append({
                "id": len(chunks),
                "title": current_title,
                "content": body,
            })

    for line in text.splitlines():  #text 是整份报告（一个长字符串）。.splitlines() 把它按换行切成列表：
        stripped = line.strip() #去掉首尾空格

        # 跳过纯分隔线
        if re.fullmatch(r"=+", stripped): # fullmatch是全部匹配的意思。  =+ 的意思是：一个或多个等号。
            continue  # 立刻结束这一轮循环，跳到下一行。

        # 判断是否分节标记
        is_big_section = re.match(r"^[一二三四五六七八九十]+、", stripped)
        #is_sub_section = re.match(r"^\d+\.\s*", stripped) and len(stripped) < 15
        is_sub_section = (
                re.match(r"^\d+\.\s*", stripped)
                and not stripped.endswith(("；", "。", ";", "，"))
        )

        if is_big_section or is_sub_section:
            flush()  # 先保存上一块
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    flush()  # 最后一块
    return chunks

# ══════════════════════════════════════════════════════════
# ② 索引构建
# ══════════════════════════════════════════════════════════
def tokenize(text):
    """把文本切成「检索单元」

    中文没有空格，简单做法：
        - 每个汉字算一个 token
        - 连续英文/数字算一个 token
        - 全部转小写

    例："空腹血糖(GLU)：7.2 mmol/L"
        → ["空", "腹", "血", "糖", "glu", "7", "2", "mmol", "l"]
    """
    text = text.lower()
    # 英文单词/数字整体保留，其余按单字切
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    return tokens


def build_index(chunks):
    """为每个 chunk 建词频表

    返回：[{"id":0, "title":..., "content":..., "tf": Counter(...)}, ...]
    """
    index = []
    for chunk in chunks:
        # 标题也参与检索（"血糖"可能只出现在标题里）
        tokens = tokenize(chunk["title"] + " " + chunk["content"])
        index.append({
            **chunk,
            "tf": Counter(tokens),
            "length": len(tokens),
        })
    return index


# ══════════════════════════════════════════════════════════
# ③ 检索（今天的核心）
# ══════════════════════════════════════════════════════════
def compute_idf(index):
    """计算每个词的 IDF（逆文档频率）

    IDF = log(总文档数 / 包含该词的文档数)

    含义：越少见的词越重要。
        "的" 出现在所有 chunk → IDF 低 → 不重要
        "血糖" 只出现在 1-2 个 chunk → IDF 高 → 重要

    返回：{词: idf值}
    """
    total = len(index)
    df = Counter()
    for chunk in index:
        for token in chunk["tf"]:
            df[token] += 1

    idf = {}
    for token, freq in df.items():
        # +1 平滑，避免除以 0
        idf[token] = math.log((total + 1) / (freq + 1)) + 1
    return idf


def score_chunk(question_tokens, chunk, idf):
    """计算一个 chunk 和问题的相关度（BM25 简化版）

    思路：
        对问题里的每个词，看它在 chunk 里出现几次（tf）
        乘以这个词的重要性（idf）
        加起来就是总分

    score = Σ (tf_in_chunk × idf_of_token)

    TODO 2: 实现这个函数
        1. 初始化 score = 0
        2. 遍历 question_tokens
        3. 对每个 token：
            - 从 chunk["tf"] 取它出现的次数，用 .get(token, 0)
            - 从 idf 取它的权重，用 .get(token, 0)
            - 相乘后累加到 score
        4. 返回 score
    """
    # TODO 2: 你的代码
    score = 0
    for token in question_tokens:
        tf = chunk["tf"].get(token, 0)
        # chunk["tf"]["我"]   KeyError: '我'
        # chunk["tf"].get("我", 0)  # ✅ 返回 0
        # 因为大部分问题里的字，在很多块里都不存在。 如果用中括号，每次查不到都会崩溃。
        weight = idf.get(token, 0) #出现得越多，IDF 越低。 compute_idf()
        score += tf * weight
    return score


def retrieve(question, index, idf, top_k=TOP_K):
    """检索最相关的 top_k 个 chunk

    返回：[{"id":..., "title":..., "content":..., "score":...}, ...]
    按 score 降序排列
    """
    question_tokens = tokenize(question) #过滤符号

    # TODO 3: 计算每个 chunk 的分数并排序
    #   提示：
    #       scored = []
    #       for chunk in index:
    #           s = score_chunk(question_tokens, chunk, idf)
    #           if s > MIN_SCORE:
    #               scored.append({... chunk, "score": s})
    #       scored.sort(key=lambda x: x["score"], reverse=True)
    #       return scored[:top_k]
    scored = []
    for chunk in index:
        s = score_chunk(question_tokens, chunk, idf)
        if s > MIN_SCORE: # 过滤掉 3.0 分的块。
            scored.append({
                "id": chunk["id"],
                "title": chunk["title"],
                "content": chunk["content"],
                "score": s,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ══════════════════════════════════════════════════════════
# ④ 拼接 Prompt
# ══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你是一个体检报告解读助手。

规则（必须严格遵守）：
1. 只能根据【参考资料】里的内容回答，不要用你自己的医学知识补充。
2. 如果参考资料里没有相关信息，直接回答「报告中未提及」，不要猜测。
3. 回答时标注信息来自哪个小节，格式：[来源：小节名]
4. 数值要原样引用，不要换算或近似。
5. 用中文回答，简洁直接。"""


def build_prompt(question, chunks):
    """把检索到的 chunk 拼成参考资料"""
    if not chunks:
        context = "（没有检索到相关内容）"
    else:
        parts = []
        for c in chunks:
            parts.append(f"【{c['title']}】\n{c['content']}")
        context = "\n\n---\n\n".join(parts)

    return f"""【参考资料】
{context}

【问题】
{question}

请根据上面的参考资料回答。"""


# ══════════════════════════════════════════════════════════
# ⑤ 调用模型
# ══════════════════════════════════════════════════════════
def call_model(api_key, system_prompt, user_prompt, retries=3):
    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "max_output_tokens": 1024,
        "temperature": 0,
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
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

    text_parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))
    return "".join(text_parts)


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def answer(api_key, question, index, idf, verbose=True):
    chunks = retrieve(question, index, idf)

    if verbose:
        print(f"\n问题：{question}")
        print(f"\n检索到 {len(chunks)} 个片段：")
        for c in chunks:
            print(f"  [{c['score']:.2f}] {c['title']}")

    prompt = build_prompt(question, chunks)
    reply = call_model(api_key, SYSTEM_PROMPT, prompt)

    if verbose:
        print(f"\n回答：{reply}")
    return reply, chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="要问的问题")
    args = parser.parse_args()

    api_key = load_api_key()

    # ①② 切分 + 建索引
    text = REPORT_FILE.read_text(encoding="utf-8")
    chunks = chunk_report(text)
    index = build_index(chunks)
    idf = compute_idf(index)

    print(f"报告切成 {len(chunks)} 个 chunk")
    for c in chunks:
        print(f"  [{c['id']}] {c['title']}  ({len(c['content'])} 字)")

    # 单问 or 批量测试
    if args.question:
        answer(api_key, args.question, index, idf)
        return

    questions = [
        "我的血糖怎么样？",
        "肝功能有没有问题？",
        "医生建议我做什么？",
        "我的血型是什么？",        # 报告里没有 → 应该回答"未提及"
    ]
    for q in questions:
        print("\n" + "=" * 60)
        answer(api_key, q, index, idf)
        print()


if __name__ == "__main__":
    main()
