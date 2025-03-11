import argparse 
import polars as pl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge datasets to a bigger one")
    parser.add_argument("--datasets", type=str, default="./datasets/origin", help="Parent path of the dataset files")
    parser.add_argument("--out", type=str, default="./datasets/merge", help="Path of the output folder")
    args = parser.parse_args()
    
    from pathlib import Path
    datasets_parent = Path(args.datasets)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    if not datasets_parent.is_dir():
        print(f"{datasets_parent} is not a directory")
        exit()
    if not out_path.is_dir():
        print(f"{out_path} is not a directory")
        exit()
    
    import re
    from datetime import datetime
    dir_cnt = 0
    datasets = []
    for dataset in datasets_parent.iterdir():
        if not dataset.is_dir(): continue
        dir_cnt += 1
        _name = re.split(r'[_-]+', dataset.name)
        try: 
            head, date, h, min, s, tail = _name
            if head != "record" and tail != "out": continue
            y, m, d = date[:4], date[4:6], date[6:]
            timestamp = datetime(*map(lambda x: int(x), [y, m, d, h, min, s]))
            datasets.append((timestamp, dataset))
        except Exception as e: 
            continue
        
    if len(datasets) == 0:
        print(f"[Error] No valid dataset found in {datasets_parent}")
        exit()
        
    print(f"[Info] {dir_cnt} folders exist in {datasets_parent}")
    print(f"[Info] {len(datasets)} valid datasets were found.")
    check = input("Press Y to continue [Y/n] >>> ")
    if check != "Y": exit()
    
    final_df = pl.DataFrame(
        schema = {
            "img_dir": pl.Utf8,
            "move": pl.Boolean,
            "angle": pl.Float64,
            "attack": pl.Boolean,
            "skill": pl.Boolean,
            "weapon": pl.Boolean,
            "source": pl.Utf8,
        })
    frame_rename_map = {}
    
    datasets: list[Path] = [x[1] for x in sorted(datasets, key=lambda x: x[0])]
    for dataset in datasets:
        # ensure the existance of CSV
        data_attr_path = dataset / "dataset.csv"
        if not data_attr_path.is_file(): 
            print(f"[Warning] Broken dataset: CSV not found, on {data_attr_path}")
            check = input("Press Y to ignore and continue [Y/n] >>> ")
            if check != "Y": exit()
            else: continue
            
        broken = False
        frames = {}
        for f in dataset.iterdir():
            if f.suffix == '.npy':
                # ensure frame cnt can be added
                try:
                    _ = int(f.stem)
                except Exception:
                    print(f"[Error] Broken dataset: {f.stem} is not a valid frame name, on {dataset}")
                    check = input("Press Y to ignore and continue [Y/n] >>> ")
                    if check != "Y": exit()
                    else: 
                        broken = True
                        break
                frames[f.name] = 1
        if broken: continue
        
        # judge if the frame in CSV is not in dataset
        df = pl.read_csv(
            data_attr_path, 
            schema = {
                "img_dir": pl.Utf8,
                "move": pl.Boolean,
                "angle": pl.Float64,
                "attack": pl.Boolean,
                "skill": pl.Boolean,
                "weapon": pl.Boolean,
            })
        for dir in df["img_dir"].to_list():
            if dir not in frames: 
                print(f"[Warning] Broken dataset: {dir} not found, on {dataset}")
                check = input("Press Y to ignore and continue [Y/n] >>> ")
                if check != "Y": exit()
                else: 
                    broken = True
                    break
            frames[dir] -= 1
        if broken: continue
        
        # judge if the frame in dataset is not in CSV
        broken = [k for k, v in frames.items() if v != 0]
        if len(broken) > 0:
            print(f"[Warning] Broken dataset: {broken} not in CSV, on {dataset}")
            check = input("Press Y to ignore and continue [Y/n] >>> ")
            if check != "Y": exit()
            else: continue
            
        # dataset is ok
        if "source" not in df.columns:
            df = df.with_columns(pl.lit(dataset.name).alias("source"))
        frame_rename_map[str(dataset.absolute())] = final_df.shape[0]
        final_df = final_df.vstack(df)
        
    # all datasets ready, start to merge
    import shutil
    from tqdm import tqdm
    def rename_and_move_child_frame(p: Path, target: Path, offset: int):
        npy_files = list(p.iterdir())
        npy_files = [f for f in npy_files if f.suffix == '.npy']
        for frame in tqdm(npy_files, desc=f"Merging {p.name}", mininterval=0.5):
            try:
                idx = int(frame.stem)
                new_path = target / f"{offset + idx}.npy"
                shutil.move(str(frame), str(new_path))
            except Exception as e:
                print(f"[FATAL] Error processing file {f.name}: {e}")
    
    print("[Info] All settled, ready to merge, please check the params:")
    print(f"[Info] \tOutput dir: {str(out_path.absolute())}")
    print(f"[Info] \tDatasets: {frame_rename_map}")
    check = input("Press Y to merge! [Y/n] >>> ")
    if check != "Y": exit()
    
    import time
    print("[Info] Start to merge, !!!do not close the terminal!!!")
    out_path = out_path / f"merged_{time.strftime('%Y%m%d-%H_%M_%S')}-_out"
    out_path.mkdir(parents=True, exist_ok=True)
    final_df = final_df.with_columns(pl.Series("img_dir", list(map(lambda x: f"{x}.npy", range(1, len(final_df)+1)))))
    final_df.write_csv(out_path / "dataset.csv")
    
    for dir_name, offset in frame_rename_map.items():
        rename_and_move_child_frame(Path(dir_name), out_path, offset)
    
    
    print("[Info] Merge finished, please check the output folder :)")