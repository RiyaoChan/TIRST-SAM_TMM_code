import argparse
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--keys", nargs="*", default=None)
    args = parser.parse_args()

    default_keys = [
        "use_asg_hq",
        "asg_variant",
        "asg_loc",
        "asg_strength_enc",
        "asg_strength_dec",
        "use_hldf",
        "hldf_layers",
        "hldf_hidden_dim",
        "hldf_use_hq_router",
        "hldf_router_temp",
        "use_amgd",
        "use_dog_amgd",
        "dog_amgd_mode",
        "amgd_routing",
        "amgd_interm_layer",
        "amgd_branch_design",
        "amgd_detail_layer",
        "amgd_structure_layer",
        "amgd_background_layer",
        "dog_amgd_strength",
        "lr_head",
        "lr_encoder",
        "weight_decay",
        "val_thr_search",
        "val_thr_min",
        "val_thr_max",
        "val_thr_step",
        "mask_suffix",
    ]
    keys = args.keys or default_keys

    for path_text in args.checkpoints:
        path = pathlib.Path(path_text)
        print(f"### {path}")
        if not path.exists():
            print("missing")
            continue
        ckpt = torch.load(path, map_location="cpu")
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        for key in keys:
            if key in ckpt_args:
                print(f"{key}={ckpt_args[key]!r}")


if __name__ == "__main__":
    main()
