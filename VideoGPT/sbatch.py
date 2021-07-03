import argparse
import subprocess


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="name")
parser.add_argument('--cpu', action='store_true', default=False,
                    help='uses CPUs')
# parser.add_argument('--lab', action='store_true', default=False,
#                     help='uses csxu')
parser.add_argument('--bigger-gpu', action='store_true', default=False,
                    help='uses K80 GPU')
parser.add_argument('--biggest-gpu', action='store_true', default=False,
                    help='uses V100 GPU')
parser.add_argument('--file', type=str, default="scripts/train_videogpt.py")
parser.add_argument('--params', type=str, default="--vqvae ucf101_stride4x4x4 --data_path datasets/ucf101/ --gpus 4")
parser.add_argument('--module', type=str, default="miniconda3/4.6.14")
args = parser.parse_args()


def slurm_script_generalized():
    return r"""#!/bin/bash
#SBATCH {}
{}
#SBATCH -p reserved --reservation=slerman-20210701
#SBATCH -t 5-00:00:00 -o ./{}.log -J {}
#SBATCH --mem=24gb 
{}
module load {}
source activate env
python3 {} {}
""".format("-c 1" if args.cpu else "-p gpu",
           "" if args.cpu else "#SBATCH -p csxu -A cxu22_lab" if False else "#SBATCH --gres=gpu:4",
           # "#SBATCH -p csxu -A cxu22_lab" if args.cpu else "#SBATCH -p csxu -A cxu22_lab --gres=gpu",
           args.name, args.name,
           "#SBATCH -C K80" if args.bigger_gpu else "#SBATCH -C V100" if args.biggest_gpu else "",
           args.module, args.file, args.params)


with open("sbatch_script", "w") as file:
    file.write(slurm_script_generalized())
subprocess.call(['sbatch {}'.format("sbatch_script")], shell=True)
