import os


def find_test_logs(root):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "test_log.txt" in filenames:
            results.append(os.path.join(dirpath, "test_log.txt"))
    return sorted(results)


def parse_test_log(path):
    miou = None
    oa = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "mIoU" in line and "allAcc" in line:
                parts = line.replace(",", " ").replace("\t", " ").split()
                nums = []
                for token in parts:
                    try:
                        val = float(token)
                        nums.append(val)
                    except ValueError:
                        pass
                if len(nums) >= 3:
                    miou = nums[0]
                    oa = nums[-1]
                    break
    return miou, oa


def main():
    exp_root = os.path.join("experiments", "point_transformer_v1_logs")
    if not os.path.isdir(exp_root):
        print(f"Experiments root not found: {exp_root}")
        return

    logs = find_test_logs(exp_root)
    if not logs:
        print(f"No test_log.txt files found under {exp_root}")
        return

    print("RealRatio\tmIoU\tOA\tPath")
    for log_path in logs:
        parts = log_path.split(os.sep)
        real_ratio = "?"
        for part in parts:
            if part.startswith("model_"):
                real_ratio = part.replace("model_", "")
                break
        miou, oa = parse_test_log(log_path)
        print(f"{real_ratio}\t{miou}\t{oa}\t{log_path}")


if __name__ == "__main__":
    main()


