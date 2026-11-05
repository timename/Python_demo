# -*- coding: utf-8 -*-
"""
诗词字数统计器
读 poem.txt，统计每个汉字出现次数，输出前 10 名到 result.txt
"""

INPUT_FILE = "poem.txt"
OUTPUT_FILE = "result.txt"

# 中文标点集合（要排除的字符）
# PUNCTUATIONS = "，。、；：？！“”‘’（）《》—…·"
PUNCTUATIONS = "，。、；：？！“”‘’（）《》—…·,.;:?!\"'()[]{}<>"

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


# def read_poem(path):
#     """读取文件内容，文件不存在时返回 None"""
#     # 你的代码：try/except 读文件
#
#     pass

def read_poem(path):
    """读取文件内容，文件不存在时返回 None"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # 这里写什么？
        # 注意：main 里已经负责 print 提示了，
        # 那这个函数里应该 print 吗？还是只 return？
        return None
        #pass

def count_chars(text):
    """统计每个汉字出现次数，返回字典"""
    # 你的代码：遍历字符，跳过标点
    content = {}
    for char in text:
        if char not in PUNCTUATIONS and char != " " and char != "\n":  #更简单写法：if not char.isspace() and char not in PUNCTUATIONS:
            # content[char] += 1 #Python 字典访问不存在的 key 会抛异常，不会自动初始化为 0。
            # 改法1：
            # if char in content:  # ← 先判断有没有
            #     content[char] += 1
            # else:
            #     content[char] = 1  # ← 没有就设成 1
            # 改法2:
            content[char] = content.get(char, 0) + 1 #get(char, 0) 的意思是：如果有 char，返回它的值；如果没有，返回 0。
            # 所以 content.get(char, 0) + 1 第一次会算出 0 + 1 = 1，正好给它设初值。
    return content  # ← 加上这行
    #pass

def count_chars_chinese(text):
    """统计每个汉字出现次数，返回字典"""
    # 你的代码：遍历字符，跳过标点
    content = {}
    for char in text:
        if is_chinese(char):
            content[char] = content.get(char, 0) + 1
    return content

# def top_n(counter, n=10):
#     """返回出现次数最多的 n 个，列表形式 [(字, 次数), ...]"""
#     # 你的代码：sorted + lambda
#     sorted_items = sorted(counter.items(), key=counter.vaule(), reverse = True)
#     return sorted_items[:n]
#     #pass

def top_n(counter, n=10):
    """返回出现次数最多的 n 个，列表形式 [(字, 次数), ...]"""
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True) #sorted默认升序
    return sorted_items[:n] #这里只返回n个值 :n切片

def write_result(result_text, path):
    """把结果写入文件"""
    # 你的代码：with open(..., "w")
    with open(path, "w", encoding="utf-8") as f:
        f.write(result_text)
    # pass

def main():
    text = read_poem(INPUT_FILE)
    if text is None:
        print("文件不存在，请检查路径")
        return

    counter = count_chars_chinese(text)
    ranking = top_n(counter, 10)

    # 组装输出文本
    lines = []
    lines.append("——《诗词》字数统计——")
    lines.append(f"总计：{sum(counter.values())} 字（不含标点）")
    lines.append("")
    lines.append("前 10 名：")
    for i, (char, cnt) in enumerate(ranking, start=1):
        lines.append(f"{i}. {char}  {cnt} 次")

    result_text = "\n".join(lines)
    print(result_text)          # 屏幕上先看一眼
    write_result(result_text, OUTPUT_FILE)
    print(f"\n已写入 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()