#!/usr/bin/env python3
"""用 DeepSeek 给 2023/2024 就业单位打标签（含行业与公开财务信息字段）。"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CACHE_PATH = OUT_DIR / "employer_labels_llm_cache.json"
RESULT_CSV = OUT_DIR / "employer_labels_llm.csv"
REVIEW_CSV = OUT_DIR / "employer_labels_review.csv"
STUDENT_LABELED_CSV = OUT_DIR / "employment_with_employer_labels.csv"

CATEGORIES = [
    "国企",
    "大型民企",
    "中小民企",
    "集体所有制",
    "外企",
    "军工",
    "部队/应征",
    "高校/升学",
    "机关事业单位/基层项目",
    "其他/无法判断",
]

ENTERPRISE_CATS = {"国企", "大型民企", "中小民企", "集体所有制", "外企", "军工"}
PRIVATE_OR_FOREIGN = {"大型民企", "中小民企", "外企"}

SYSTEM_PROMPT = f"""你是熟悉中国就业单位与企业所有制分类的分析助手。
请根据单位名称（可结合常见公开信息）判断类别，并尽量补充行业与公开财务/实控人信息。

【类别必须且只能从下列选择一个】
{chr(10).join('- ' + c for c in CATEGORIES)}

【字段要求】
1. 所有单位都必须输出：单位名称、类别、置信度（高/中/低）、理由（一句话）。
2. 若类别属于「国企、大型民企、中小民企、集体所有制、外企、军工」，额外尽量输出：
   - 行业
   - 最近年度总资产
   - 净资产
   - 年度净利润
   - 净现金流
   - 财务数据对应年份（如 2023）
3. 若类别属于「大型民企、中小民企、外企」，额外尽量输出：
   - 实控人姓名
   - 实控人国籍
4. 若类别属于「部队/应征、高校/升学、机关事业单位/基层项目、其他/无法判断」：
   - 财务与实控人字段一律填「不适用」
5. 不确定的信息必须填「未知」，禁止编造具体数字或人名。
6. 数字尽量用中文常见写法（如「约120亿元」）；完全不知道就写「未知」。

【输出格式】
只输出一个 JSON 对象，不要 Markdown，不要其它说明。格式如下：
{{
  "items": [
    {{
      "单位名称": "...",
      "类别": "...",
      "置信度": "高|中|低",
      "理由": "...",
      "行业": "...",
      "最近年度总资产": "...",
      "净资产": "...",
      "年度净利润": "...",
      "净现金流": "...",
      "财务数据年份": "...",
      "实控人姓名": "...",
      "实控人国籍": "..."
    }}
  ]
}}
"""


def parse_employment_xls(path: Path) -> pd.DataFrame:
    """就业 xls 实际是 HTML 表格。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.I | re.S)
    cells_all = []
    for r in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=re.I | re.S)
        cells = [re.sub(r"<[^>]+>", "", c).replace("\xa0", " ").strip() for c in cells]
        if cells:
            cells_all.append(cells)
    if not cells_all:
        raise ValueError(f"无法解析表格: {path}")
    header = cells_all[0]
    data_rows = [c for c in cells_all[1:] if len(c) == len(header)]
    return pd.DataFrame(data_rows, columns=header)


def load_unique_units() -> pd.DataFrame:
    records = []
    for year, fname in [(2023, "2023年就业数据.xls"), (2024, "2024年就业数据.xls")]:
        df = parse_employment_xls(DATA_DIR / fname)
        for _, row in df.iterrows():
            name = str(row.get("单位名称", "") or "").strip()
            dest = str(row.get("毕业去向", "") or "").strip()
            if not name:
                continue
            records.append({"年份": year, "单位名称": name, "毕业去向": dest})
    long_df = pd.DataFrame(records)
    # 去重单位：保留出现年份列表
    grouped = (
        long_df.groupby("单位名称", as_index=False)
        .agg(
            出现年份=("年份", lambda s: ",".join(str(x) for x in sorted(set(s)))),
            出现次数=("单位名称", "size"),
            毕业去向样例=("毕业去向", lambda s: " / ".join(sorted({x for x in s if x})[:3])),
        )
    )
    return grouped.sort_values("单位名称").reset_index(drop=True), long_df


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_array(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        if not isinstance(data, list):
            raise ValueError("JSON 根节点不是数组")
        return data
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            raise
        return json.loads(m.group(0))


def normalize_item(raw: dict, fallback_name: str | None = None) -> dict:
    name = str(raw.get("单位名称") or fallback_name or "").strip()
    cat = str(raw.get("类别") or "其他/无法判断").strip()
    if cat not in CATEGORIES:
        # 容错映射
        for c in CATEGORIES:
            if c in cat or cat in c:
                cat = c
                break
        else:
            cat = "其他/无法判断"
    conf = str(raw.get("置信度") or "低").strip()
    if conf not in {"高", "中", "低"}:
        conf = "低"

    def g(key: str, default: str = "未知") -> str:
        v = raw.get(key)
        if v is None:
            return default
        s = str(v).strip()
        return s if s else default

    item = {
        "单位名称": name,
        "类别": cat,
        "置信度": conf,
        "理由": g("理由", ""),
        "行业": g("行业"),
        "最近年度总资产": g("最近年度总资产"),
        "净资产": g("净资产"),
        "年度净利润": g("年度净利润"),
        "净现金流": g("净现金流"),
        "财务数据年份": g("财务数据年份"),
        "实控人姓名": g("实控人姓名"),
        "实控人国籍": g("实控人国籍"),
    }

    if cat not in ENTERPRISE_CATS:
        for k in ["行业", "最近年度总资产", "净资产", "年度净利润", "净现金流", "财务数据年份", "实控人姓名", "实控人国籍"]:
            if item[k] in {"未知", ""}:
                item[k] = "不适用"
    elif cat not in PRIVATE_OR_FOREIGN:
        if item["实控人姓名"] in {"未知", ""}:
            item["实控人姓名"] = "不适用"
        if item["实控人国籍"] in {"未知", ""}:
            item["实控人国籍"] = "不适用"
    return item


def heuristic_prelabel(name: str, dest: str) -> dict | None:
    """对明显非企业场景做轻量预标，节省调用；仍可被缓存覆盖。"""
    if dest in {"境内升学"} or any(k in name for k in ("大学", "学院", "研究生")):
        return normalize_item(
            {
                "单位名称": name,
                "类别": "高校/升学",
                "置信度": "高",
                "理由": "毕业去向或单位名称指向升学/高校",
            }
        )
    if dest in {"应征义务兵"} or name in {"火箭军"} or "部队" in name:
        return normalize_item(
            {
                "单位名称": name,
                "类别": "部队/应征",
                "置信度": "高",
                "理由": "毕业去向或单位名称指向部队/应征",
            }
        )
    if dest in {"国家基层项目", "地方基层项目"}:
        return normalize_item(
            {
                "单位名称": name,
                "类别": "机关事业单位/基层项目",
                "置信度": "高",
                "理由": "毕业去向为基层项目",
            }
        )
    if dest == "待就业" and (not name or name in {"无", "暂无", "待定"}):
        return normalize_item(
            {
                "单位名称": name or "待就业",
                "类别": "其他/无法判断",
                "置信度": "高",
                "理由": "待就业且无有效单位",
            }
        )
    return None


def call_deepseek(client: OpenAI, model: str, batch_names: list[str], retries: int = 3) -> list[dict]:
    user_prompt = (
        "请对下列就业单位逐一打标，严格按系统要求输出 JSON 对象（含 items 数组）。单位列表：\n"
        + "\n".join(f"{i+1}. {n}" for i, n in enumerate(batch_names))
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            # deepseek json_object 可能包一层 {"items":[...]} 或 {"results":[...]}
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for key in ("items", "results", "data", "单位列表", "labels"):
                        if key in parsed and isinstance(parsed[key], list):
                            parsed = parsed[key]
                            break
                    else:
                        # 单对象
                        if "单位名称" in parsed:
                            parsed = [parsed]
                        else:
                            parsed = extract_json_array(content)
                elif not isinstance(parsed, list):
                    parsed = extract_json_array(content)
            except json.JSONDecodeError:
                parsed = extract_json_array(content)

            out = []
            for i, raw in enumerate(parsed):
                if not isinstance(raw, dict):
                    continue
                fb = batch_names[i] if i < len(batch_names) else None
                out.append(normalize_item(raw, fb))
            # 按名称对齐缺失项
            by_name = {x["单位名称"]: x for x in out if x.get("单位名称")}
            aligned = []
            for n in batch_names:
                if n in by_name:
                    aligned.append(by_name[n])
                else:
                    # 模糊匹配
                    hit = next((v for k, v in by_name.items() if n in k or k in n), None)
                    if hit:
                        hit = dict(hit)
                        hit["单位名称"] = n
                        aligned.append(hit)
                    else:
                        aligned.append(
                            normalize_item(
                                {
                                    "单位名称": n,
                                    "类别": "其他/无法判断",
                                    "置信度": "低",
                                    "理由": "模型未返回该单位，需人工复核",
                                }
                            )
                        )
            return aligned
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DeepSeek 调用失败: {last_err}")


def disable_local_proxies() -> None:
    """不使用本机代理（尤其 10808），直连 DeepSeek。"""
    keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for key in keys:
        os.environ.pop(key, None)


def build_client() -> tuple[OpenAI, str]:
    disable_local_proxies()
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
    model = os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL") or "deepseek-chat"
    if not api_key or api_key.startswith("sk-你的"):
        raise SystemExit(
            "未检测到有效 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入密钥后再运行。"
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def export_tables(cache: dict, units_meta: pd.DataFrame, long_df: pd.DataFrame) -> None:
    rows = []
    for _, meta in units_meta.iterrows():
        name = meta["单位名称"]
        item = cache.get(name)
        if not item:
            continue
        row = {
            **item,
            "出现年份": meta["出现年份"],
            "出现次数": meta["出现次数"],
            "毕业去向样例": meta["毕业去向样例"],
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    review = result[result["置信度"].isin(["低", "中"]) | (result["类别"] == "其他/无法判断")].copy()
    review.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    # 回填到学生就业明细
    label_cols = [
        "类别",
        "置信度",
        "理由",
        "行业",
        "最近年度总资产",
        "净资产",
        "年度净利润",
        "净现金流",
        "财务数据年份",
        "实控人姓名",
        "实控人国籍",
    ]
    label_df = result[["单位名称"] + label_cols].drop_duplicates("单位名称")
    merged = long_df.merge(label_df, on="单位名称", how="left")
    merged.to_csv(STUDENT_LABELED_CSV, index=False, encoding="utf-8-sig")
    print(f"已写出: {RESULT_CSV}")
    print(f"已写出: {REVIEW_CSV}（待复核 {len(review)} 条）")
    print(f"已写出: {STUDENT_LABELED_CSV}")
    if len(result):
        print("类别分布:")
        print(result["类别"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 就业单位打标")
    parser.add_argument("--batch-size", type=int, default=8, help="每批单位数")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个未缓存单位（调试用）")
    parser.add_argument("--sleep", type=float, default=0.8, help="批次间隔秒")
    parser.add_argument("--export-only", action="store_true", help="只根据缓存导出表格")
    parser.add_argument("--no-heuristic", action="store_true", help="禁用升学/部队等启发式预标")
    args = parser.parse_args()

    disable_local_proxies()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units_meta, long_df = load_unique_units()
    cache = load_cache()
    print(f"去重单位数: {len(units_meta)}，已缓存: {len(cache)}")

    if args.export_only:
        export_tables(cache, units_meta, long_df)
        return

    # 建立单位 -> 去向样例，便于启发式
    dest_map = dict(zip(units_meta["单位名称"], units_meta["毕业去向样例"]))

    pending = [n for n in units_meta["单位名称"].tolist() if n not in cache]
    if not args.no_heuristic:
        still = []
        for n in pending:
            pre = heuristic_prelabel(n, dest_map.get(n, ""))
            if pre:
                cache[n] = pre
            else:
                still.append(n)
        save_cache(cache)
        pending = still
        print(f"启发式预标后待调用: {len(pending)}")

    if args.limit and args.limit > 0:
        pending = pending[: args.limit]

    if pending:
        client, model = build_client()
        print(f"使用模型: {model}，待打标: {len(pending)}")
        for i in range(0, len(pending), args.batch_size):
            batch = pending[i : i + args.batch_size]
            print(f"批次 {i // args.batch_size + 1}: {len(batch)} 家 ...")
            labeled = call_deepseek(client, model, batch)
            for item in labeled:
                cache[item["单位名称"]] = item
            save_cache(cache)
            time.sleep(args.sleep)
    else:
        print("无待打标单位，直接导出。")

    export_tables(cache, units_meta, long_df)


if __name__ == "__main__":
    main()
