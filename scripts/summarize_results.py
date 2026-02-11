"""
汇总所有实验结果, 生成对比表格
扫描 outputs/ 下所有 results.json 和 eval_results.json, 汇总为一张表
"""

import os
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_results():
    """收集所有实验结果"""
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    results = []

    # 主模型结果
    main_eval = os.path.join(outputs_dir, "eval_results.json")
    if os.path.exists(main_eval):
        with open(main_eval) as f:
            data = json.load(f)
        results.append({
            "model": "DINOv3-LoRA-Mask2Former (Ours)",
            "type": "main",
            **_extract(data),
        })

    # 基线和消融结果
    for json_path in sorted(glob.glob(os.path.join(outputs_dir, "*", "results.json"))):
        with open(json_path) as f:
            data = json.load(f)
        dirname = os.path.basename(os.path.dirname(json_path))
        model_name = data.get("model", data.get("experiment", dirname))
        exp_type = "baseline" if "baseline" in dirname else "ablation"
        results.append({
            "model": model_name,
            "type": exp_type,
            **_extract(data),
        })

    return results


def _extract(data):
    """从结果 JSON 中提取关键指标"""
    info = {
        "params": data.get("trainable_params", data.get("total_params", "?")),
        "time": data.get("training_time", "?"),
    }
    for domain in ["bijie_val", "moxizhen_test"]:
        if domain in data:
            d = data[domain]
            prefix = "val" if "bijie" in domain else "test"
            info[f"{prefix}_miou"] = d.get("miou", 0)
            info[f"{prefix}_f1"] = d.get("f1", 0)
            info[f"{prefix}_precision"] = d.get("precision", 0)
            info[f"{prefix}_recall"] = d.get("recall", 0)
    return info


def print_table(results):
    """打印对比表格"""
    if not results:
        print("No results found in outputs/")
        return

    print("\n" + "=" * 100)
    print("  实验结果汇总")
    print("=" * 100)

    header = f"{'Model':<35} {'Type':<10} {'Val mIoU':>9} {'Val F1':>8} {'Test mIoU':>10} {'Test F1':>9} {'Params':>12}"
    print(header)
    print("-" * 100)

    for r in results:
        print(f"{r['model']:<35} {r['type']:<10} "
              f"{r.get('val_miou', 0):>9.4f} {r.get('val_f1', 0):>8.4f} "
              f"{r.get('test_miou', 0):>10.4f} {r.get('test_f1', 0):>9.4f} "
              f"{r.get('params', '?'):>12}")

    print("=" * 100)


def save_summary(results):
    """保存汇总到 JSON"""
    output_path = os.path.join(PROJECT_ROOT, "outputs", "experiment_summary.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {output_path}")


def main():
    results = collect_results()
    print_table(results)
    if results:
        save_summary(results)


if __name__ == "__main__":
    main()
