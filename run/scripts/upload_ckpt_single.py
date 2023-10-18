import os
import click
import json
import shutil
from typing import Any

class EasyDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

@click.command()
@click.option('--ckpt_dir', help='Checkpoint dir', metavar='[ZIP|DIR]',type=str, default='/home/tiger/code/checkpoints')
@click.option('--tmp_dir', help='temp dir', metavar='[ZIP|DIR]',type=str, default='/home/tiger/code/upload')
@click.option('--upload', help='whether to upload to hdfs',type=bool, default=False)
def main(**kwargs):
    # Initialize config.
    opts = EasyDict(kwargs) # Command line arguments.
    os.makedirs(opts.tmp_dir, exist_ok=True)
    ckpt_dir = opts.ckpt_dir

    file_to_save = ['metric-fid50k.jsonl', 'log.txt', 'stats.jsonl', 'training_options.json']
    metric_path = os.path.join(ckpt_dir,'metric-fid50k.jsonl')
    # select checkpoint with best fid
    with open(metric_path , 'r') as f:
        metrics = list(map(json.loads, f))
    metrics_sorted = sorted(metrics, key=lambda d: d['results']['fid50k'])
    snapshot_pkl = metrics_sorted[0]['snapshot_pkl'].replace(".pkl", ".pt")
    file_to_save.append(snapshot_pkl)
    # select tf files
    tf_file = [filename for filename in os.listdir(ckpt_dir) if filename.startswith("events.out.tfevents")]
    file_to_save += tf_file
    # upload configs, logs and ckpts
    for file_name in file_to_save:
        shutil.copy(os.path.join(ckpt_dir,file_name), os.path.join(opts.tmp_dir,file_name))
    
    if opts.upload:
        command = f'hdfs dfs -put {opts.tmp_dir} HDFS_PATH'
        print(command)
        os.system(command)

if __name__ == "__main__":
    """
    python3 run/scripts/upload_ckpt_single.py --ckpt_dir ./checkpoints/thu_res512_no_norm/00000-stylegan2-thuman_1024x1024_2023-02-06-gpus8-batch32-gamma10/ --tmp_dir ~/checkpoints/thu_res512_no_norm --upload True
    """
    main()