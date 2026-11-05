# Study_Python

Python 学习练习仓库。按周记录每阶段的学习内容与产出。

## 目录

```
Study_Python/
├── read_plus.py     # 第1周：诗词字数统计器
├── poem.txt         # 输入数据
└── .gitignore
```

## 第 1 周 · 诗词字数统计器（`read_plus.py`）

**目标**：练习文件读写、异常处理、字典计数、排序。

**功能**：读取 `poem.txt`，统计每个汉字出现次数，按次数降序输出前 10 名到 `result.txt`。

**运行**：

```bash
python read_plus.py
```

**涉及的知识点**：

| 知识点 | 用法 |
|---|---|
| `try / except FileNotFoundError` | 文件不存在时返回 `None`，不崩溃 |
| `with open(..., encoding="utf-8")` | 自动关闭文件，正确处理中文 |
| `dict.get(key, 0) + 1` | 字典计数，避免 KeyError |
| `sorted(..., key=lambda x: x[1], reverse=True)` | 按值降序排序 |
| `list[:n]` | 切片取前 n 个 |
| Unicode 范围判断 | `'\u4e00' <= char <= '\u9fff'` 只识别汉字 |

**踩过的坑**：

1. **字典 `+= 1` 报 KeyError** —— Python 字典访问不存在的 key 会抛异常，不会自动初始化为 0，必须用 `.get(key, 0)`。
2. **黑名单不可靠** —— 最初用「排除标点」的方案，结果漏掉了半角逗号 `,`，导致逗号排到第 1 名。改为白名单（只认汉字 Unicode 区间）后解决。
   > 结论：排除法永远补不全，只保留目标字符才可靠。
3. **函数漏写 `return`** —— `count_chars` 忘记返回，导致下游拿到 `None`，报 `AttributeError: 'NoneType' object has no attribute 'items'`。

**输出示例**：

```
——《诗词》字数统计——
总计：27 字（不含标点）

前 10 名：
1. 我  1 次
2. 欲  1 次
...
```
