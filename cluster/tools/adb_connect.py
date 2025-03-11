from pathlib import Path
import json
import subprocess
import argparse

def println(*args, level="info", **kwargs):
    match level:
        case "info":
            color = "\033[92m"
            args = ["Info-> ", *args]
        case "warn":
            color = "\033[93m"
            args = ["Warn-> ", *args]
        case "fatal":
            color = "\033[91m"
            args = ["Error->", *args]
        case _:
            color = "\033[0m"
    print(color, *args, "\033[0m", **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/node.json")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_file():
        print("[ADB Node Config not Found!")
        exit(1)
        
    configs = json.loads(Path(args.config).read_text())
    configs = list(map(lambda cfg: {"name": cfg["name"], "addr": cfg["iden"]}, configs))
    
    if not configs:
        println("No configs found!", level="fatal")
        exit(1)
    
    println(f"{len(configs)} Nodes are Found in {str(config_path.absolute())}: ", level="info")
    for config in configs:
        print(f"\t\--> {config['name']}: {config['addr']}")
    
    println("Enter Y to confirm and continue [Y/n]", level="warning", end="")
    if input().lower() != "y":
        println("Aborting!", level="fatal")
        exit(1)
    
    for config in configs:
        println(f"Connecting to {config['name']}")
        subprocess.run(["adb", "connect", config["addr"]])
        println(f"Connected to {config['name']}")

    println("All Done!", level="info")
