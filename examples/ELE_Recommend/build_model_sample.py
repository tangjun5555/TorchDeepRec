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


def to_list_int_fea(value: str, key: str, sep=";", vocab_size: int=100*10000) -> str:
    split = value.strip().split(sep)
    res = []
    for x in split:
        res.append(to_int_fea(x, key, vocab_size))
    return sep.join([str(x) for x in res])


def limit_list_length(value: str, max_length: int, sep=";"):
    split = value.strip().split(sep)
    res = split[:max_length]
    while len(res) < max_length:
        res.append("0")
    res = sep.join(res)
    if res.startswith(sep):
        res = "0" + res
    return res


columns = [
        "label",
        "user_id", "gender", "visit_city", "avg_price", "is_supervip", "ctr_30", "ord_30", "total_amt_30",
        "shop_id", "item_id", "city_id", "district_id", "shop_aoi_id", "shop_geohash_6", "shop_geohash_12", "brand_id", "category_1_id", "merge_standard_food_id", "rank_7", "rank_30", "rank_90",
        "shop_id_list", "item_id_list", "category_1_id_list", "merge_standard_food_id_list", "brand_id_list", "price_list", "shop_aoi_id_list", "shop_geohash6_list", "timediff_list", "hours_list", "time_type_list", "weekdays_list",
        "times", "hours", "time_type", "weekdays", "weekdays"
]


if __name__ == '__main__':
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
            split = line.strip().split(",")
            assert len(split) == 39, (split, len(split))
            label = split[0]
            assert label in ["0", "1"]

            res = [
                int(label),
                to_int_fea(split[columns.index("user_id")], "user_id"),
                int(split[columns.index("gender")]) if split[columns.index("gender")] and int(split[columns.index("gender")])>0 else 0,
                int(split[columns.index("visit_city")]) if split[columns.index("visit_city")] else 0,
                int(split[columns.index("is_supervip")]) if split[columns.index("is_supervip")] else 0,

                to_int_fea(split[columns.index("shop_id")], "shop_id"),
                to_int_fea(split[columns.index("item_id")], "item_id"),
                int(split[columns.index("category_1_id")]),
                int(split[columns.index("brand_id")]) if split[columns.index("brand_id")] else 0,

                limit_list_length(to_list_int_fea(split[columns.index("shop_id_list")], "shop_id_list"), 20),
                limit_list_length(to_list_int_fea(split[columns.index("item_id_list")], "item_id_list"), 20),
                limit_list_length(split[columns.index("category_1_id_list")], 20),
                limit_list_length(split[columns.index("brand_id_list")], 20),

                int(split[columns.index("hours")]),
                to_int_fea(split[columns.index("time_type")], "time_type", vocab_size=100),
            ]

            if idx % 10000 == 0:
                print(res[-6], len(res[-6].split(";")))
                print(res[-5], len(res[-5].split(";")))
                print(res[-4], len(res[-4].split(";")))
                print(res[-3], len(res[-3].split(";")))

            fout.write(",".join([str(x) for x in res]) + "\n")
            if idx % 10000 == 0:
                print(f"已处理{idx}行，当前行:{line.strip()}")
