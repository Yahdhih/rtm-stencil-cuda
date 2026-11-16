#!/bin/bash
# Stencil RTM d'ordre 8 : versions GPU (naive, tuiles en memoire partagee) et CPU Grace
#   sbatch slurm/bench.slurm
#SBATCH -J rtm
#SBATCH -p short
#SBATCH -A r260073
#SBATCH --constraint=armgpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o results/bench_%j.log

module load nvhpc/nvhpc-nompi/24.11 >/dev/null 2>&1
OUT=results/raw_${SLURM_JOB_ID}.csv
REPS=5
echo "version,nx,ny,nz,steps,rep,time_s,mpts_per_s,gbps_eff" > $OUT
./bin/rtm_gpu check 128 64 96 40 1 > results/check_${SLURM_JOB_ID}.txt
for n in 256 512 768; do
  for v in naive shared sharedp shared-32x4 shared-32x16 shared-64x4 shared-64x8 shared-zs shared-64x4-zs; do
    ./bin/rtm_gpu $v $n $n $n 100 $REPS >> $OUT
  done
done
# CPU : 72 coeurs d'un Grace, puis 1, 8, 36 pour la courbe de scalabilite
for cv in flat blocked; do
  for t in 1 8 18 36 72; do
    OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores ./bin/rtm_cpu $cv 512 512 512 10 $REPS >> $OUT
  done
done
echo "termine $(date)"
