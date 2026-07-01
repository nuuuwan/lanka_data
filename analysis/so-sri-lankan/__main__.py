import json

from lanka_data import CommandRunner


def computer_error(lk_pct_values, region_pct_values):
    error2_sum = 0
    for k, v in lk_pct_values.items():
        region_v = region_pct_values.get(k, 0)
        diff = region_v - v
        error2_sum += diff**2
    error = error2_sum**0.5
    return error


def main():
    N_DISPLAY = 10
    region_type = "pd"
    what_list = [
        "Ethnicity",
        "Religion",
        "Gender",
        "AgeGroup",
        #
        "Structure",
        "Walls",
        "Floor",
        "Roof",
        #
        "Water",
        "Fuel",
        "Lighting",
        "Toilet",
    ]
    n_whats = len(what_list)

    region_id_to_rank_sum = {}
    region_id_to_name = {}

    for what in what_list:
        print("-" * 32)
        print(what)
        print("-" * 32)

        cmd_lk = f"{what}/2024/LK/JSON"
        lk_pct_values = CommandRunner.run(cmd_lk)["result"][0]["pct_values"]
        print(lk_pct_values)
        cmd = f"{what}/2024/LK:{region_type}/JSON"
        error_info_list = []
        for row in CommandRunner.run(cmd)["result"]:
            region_id = row["region_id"]
            region_name = row["region_name"]
            region_pct_values = row["pct_values"]
            error = computer_error(lk_pct_values, region_pct_values)
            error_info = dict(
                region_id=region_id,
                region_name=region_name,
                error=error,
            )
            error_info_list.append(error_info)

        error_info_list.sort(key=lambda x: x["error"])
        for rank, error_info in enumerate(error_info_list, start=1):
            region_id = error_info["region_id"]
            region_name = error_info["region_name"]

            if region_id not in region_id_to_rank_sum:
                region_id_to_rank_sum[region_id] = 0
                region_id_to_name[region_id] = region_name
            region_id_to_rank_sum[region_id] += rank

            if rank <= N_DISPLAY:
                print(
                    f"#{rank+1} {region_id} {region_name} {error_info['error']:.4f}"
                )

    print("=" * 32)
    print(f"ALL (Average Rank across {n_whats} features)")
    print("=" * 32)
    for rank, (region_id, rank_sum) in enumerate(
        list(sorted(region_id_to_rank_sum.items(), key=lambda x: x[1]))[
            :N_DISPLAY
        ],
        start=1,
    ):
        region_name = region_id_to_name.get(region_id, "")
        avg_rank = rank_sum / n_whats
        print(f"#{rank} {region_id} {region_name} {avg_rank:.1f}")


if __name__ == "__main__":
    main()
