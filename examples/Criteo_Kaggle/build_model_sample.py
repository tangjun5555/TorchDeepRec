# -*- coding: utf-8 -*-

import argparse


def to_float_fea(value: str) -> float:
    try:
        return round(float(value) ** 0.5, 1)
    except Exception:
        return 0.0


def to_int_fea(value: str, key: str, vocab_size: int=100*10000) -> int:
    if not value:
        return 0
    res = hash(value + key) % (vocab_size - 1)
    assert res >= 0
    res += 1
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--res_file",
        type=str,
        required=True,
    )
    args, extra_args = parser.parse_known_args()

    with open(args.raw_file, mode="r") as fin, open(args.res_file, mode="w") as fout:
        idx = 0
        for line in fin:
            idx += 1
            split = line.strip("\n").split("\t")
            assert len(split) == 40, (split, len(split))
            res = [int(split[0])]
            for i in range(1, 14):
                res.append(to_float_fea(split[i]))
            for i in range(14, 40):
                res.append(to_int_fea(split[i], "c" + str(i)))
            fout.write(",".join([str(x) for x in res]) + "\n")
            if idx % 10000 == 0:
                print(f"已处理{idx}行，当前行:{line.strip()}")
