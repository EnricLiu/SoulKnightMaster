import json
import argparse
from pathlib import Path

from psd_tools import PSDImage
from psd_tools.api.layers import Layer

def get_layer(psd_file, layer_name):
    for layer in PSDImage.open(psd_file)._layers:
        # if layer is invisible, continue
        if not layer.visible: continue
        if layer.name == layer_name:
            return layer
            
def read_area(layer: Layer):
    ret_dict = {}
    if layer.is_group():
        ret_dict[layer.layer_id] = {
            "name": layer.name,
            "type": layer.kind,
            "area": layer.bbox,
            "item": {},
        }
        for sub_layer in iter(layer):
            sub_dict = read_area(sub_layer)
            ret_dict[layer.layer_id]["item"][sub_layer.layer_id] = sub_dict[sub_layer.layer_id]
    else:
        ret_dict[layer.layer_id] = {
            "name": layer.name,
            "type": layer.kind,
            "area": layer.bbox,
            "item": False,
        }
    return ret_dict

def read_all_area_by_groups(psd_layers: list[Layer], group_names: list[str]):
    results = []
    failed_group_names = []
    for group_name in group_names:
        error_name = "Group Name Not Found"
        for layer in psd_layers:
            if layer.name == group_name:
                target_layer = layer
                try:
                    res = read_area(target_layer)
                    results.append(res)
                    error_name = None
                except Exception as e:
                    error_name = e
                finally:
                    break
        if error_name is not None:
            failed_group_names.append((group_name, error_name))
        
    return results, failed_group_names
    
def read_areas(psd_path: Path, group_names: list[str]):
    if not psd_path.exists() or not psd_path.suffix == ".psd":
        raise ValueError(f"{psd_path} is not exists or not .psd file")
    psd_layers = PSDImage.open(psd_path)._layers
    return read_all_area_by_groups(psd_layers, group_names)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", type=str, default="./res/target.psd")
    argparser.add_argument("-o", "--output", type=str, default="./res/result.json")
    argparser.add_argument("--target_groups", nargs="+", required=True)
    args = argparser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"input file {input_path} not exists")
        exit()
    if not input_path.suffix == ".psd":
        print(f"input file {input_path} is not .psd file")
        exit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    psd_top_layers = PSDImage.open(input_path)._layers
    group_names = args.target_groups
    
    print(f"[INFO] Target PSD File: {input_path}")
    print(f"[INFO] Target Groups: {group_names}")
    print(f"[INFO] Output Path: {output_path}")
    print("[INFO] Start Processing...")
    
    results, errs = read_all_area_by_groups(psd_top_layers, group_names)
    
    for name, err in errs:
        print(f"[Error] {err}: {name}")
        
    if not results:
        print("[Error] No Target Group Found.")
        exit()
        
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Process Finished, File saved to path {output_path.absolute()}")
    except Exception as e:
        print(e)