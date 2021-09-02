#!/bin/sh
#$ -cwd
#$ -l s_gpu=1
#$ -l h_rt=4:00:00
. /etc/profile.d/modules.sh
module load python/3.6.5
module load cuda/10.0.130
pip install -r requirements.txt
pip install dgl-cu100
sh impl.sh