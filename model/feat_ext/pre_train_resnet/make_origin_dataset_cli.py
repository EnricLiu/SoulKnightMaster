import argparse
from pathlib import Path
from tools.replay_recorder.replay import make_dataset
    
# if __name__ == '__main__':
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ptlflow.get_model('liteflownet2', ckpt_path='sintel').to(DEVICE)
    
#     make_dataset(model, Path('screen.mp4'), Path('./output'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make dataset from replay file")
    parser.add_argument("--replay", type=str, required=True, nargs='+', help="Path(s) of the replay file(s)")
    parser.add_argument("--out", type=str, default="./datasets/origin", help="Path of the output folder")
    
    args = parser.parse_args()
    
    try:
        replays = map(lambda x: Path(x), args.replay)
        out_path = Path(args.out)
    except Exception as e:
        print(f"Invalid path! {e}")
    
    
    for replay in replays:
        make_dataset(replay, out_path / replay.name)