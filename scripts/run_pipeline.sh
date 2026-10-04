#!/usr/bin/env bash
# Run from any directory; all artifact paths remain relative to the repo root.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

stage=${1:-help}
dry_run=0
if [[ $# -gt 0 ]]; then shift; fi
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
if [[ $# -gt 0 ]]; then
  printf 'Unexpected argument: %s\n' "$1" >&2
  exit 2
fi
mpi_ranks=${MPI_RANKS:-16}
python_bin=${PYTHON_BIN:-python}
if [[ ! $mpi_ranks =~ ^[1-9][0-9]*$ ]]; then
  printf 'MPI_RANKS must be a positive integer.\n' >&2
  exit 2
fi

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ $dry_run -eq 0 ]]; then "$@"; fi
}
mpi() { run mpirun -np "$mpi_ranks" "$python_bin" -m mpi4py "$@"; }
fresh() {
  if [[ $dry_run -eq 0 && -e $1 ]]; then
    printf 'Existing training output: %s. Use a fresh checkout or preserve/move the prior run first.\n' "$1" >&2
    exit 1
  fi
}
setup_stage() {
  fresh b3_output
  fresh c1_output
  run "$python_bin" setup_dataset_embeddings.py --full --skip_siglip
  mpi design_baseline_fresnel.py
}
focus_stage() {
  for seed in 0 1 2; do fresh "c2_output/seed${seed}"; done
  for seed in 0 1 2; do
    mpi design_focus_opt.py --seed "$seed" --iter 200 --init random --lr 0.05
  done
}
warm_stage() {
  for seed in 0 1 2; do fresh "c3_output/100iter_seed${seed}"; done
  for seed in 0 1 2; do
    mpi design_codesign_vlm.py --phase 100iter --seed "$seed" --init c2_warmstart \
      --init-checkpoint "c2_output/seed${seed}/checkpoint_iter0100.npz" --lr 0.01 --batch_size 16
  done
}
cold_stage() {
  for seed in 0 1 2; do fresh "c3_output/100iter_seed${seed}_rand"; done
  for seed in 0 1 2; do
    mpi design_codesign_vlm.py --phase 100iter --seed "$seed" --init random --lr 0.05 --batch_size 16
  done
}
eval_stage() {
  mpi eval_zeroshot.py --design_paths \
    c1 c1_output/rho_design.npy \
    c2_seed0 c2_output/seed0/c2_final.npz \
    c2_seed1 c2_output/seed1/c2_final.npz \
    c2_seed2 c2_output/seed2/c2_final.npz \
    c2_s0_iter100 c2_output/seed0/checkpoint_iter0100.npz \
    c2_s1_iter100 c2_output/seed1/checkpoint_iter0100.npz \
    c2_s2_iter100 c2_output/seed2/checkpoint_iter0100.npz \
    c3_seed0 c3_output/100iter_seed0_rand/c3_final.npz \
    c3_seed1 c3_output/100iter_seed1_rand/c3_final.npz \
    c3_seed2 c3_output/100iter_seed2_rand/c3_final.npz \
    c3w_iter100 c3_output/100iter_seed0/c3_final.npz \
    c3w_s1_final c3_output/100iter_seed1/c3_final.npz \
    c3w_s2_final c3_output/100iter_seed2/c3_final.npz
}
transfer_stage() {
  mpi eval_transfer.py --datasets imagenet100 cifar100 food101 --vlms clip siglip dinov2 \
    --out_dir eval_output_transfer --design_paths \
    Fresnel c1_output/rho_design.npy \
    Focus_s0 c2_output/seed0/c2_final.npz \
    Focus_s1 c2_output/seed1/c2_final.npz \
    Focus_s2 c2_output/seed2/c2_final.npz \
    VLMwarm_s0 c3_output/100iter_seed0/c3_final.npz \
    VLMwarm_s1 c3_output/100iter_seed1/c3_final.npz \
    VLMwarm_s2 c3_output/100iter_seed2/c3_final.npz \
    VLMcold_s0 c3_output/100iter_seed0_rand/c3_final.npz \
    VLMcold_s1 c3_output/100iter_seed1_rand/c3_final.npz \
    VLMcold_s2 c3_output/100iter_seed2_rand/c3_final.npz
}
case "$stage" in
  main) setup_stage; focus_stage; warm_stage; cold_stage; eval_stage ;;
  setup|focus|warm|cold|eval|transfer) "${stage}_stage" ;;
  help|-h|--help)
    printf 'Usage: bash scripts/run_pipeline.sh STAGE [--dry-run]\n'
    printf 'Stages: main, setup, focus, warm, cold, eval, transfer\n'
    printf 'Environment: MPI_RANKS (default 16), PYTHON_BIN (default python)\n'
    ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
