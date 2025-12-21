#!/bin/bash
#
# run_arms.sh -- launch a set of training arms on this host, detached.
#
#   scripts/run_arms.sh                  # every arm in the table below
#   scripts/run_arms.sh tokens_point_s0  # just one
#
# The arms live in this file rather than on a shell command line, and that is
# the point. A sweep launched from an inline `ssh ... 'for x in ...'` lost its
# `model.layers=4` to a heredoc that expanded on the wrong side of the
# connection: two arms trained at two layers while claiming to be a four-layer
# seed sweep, and nothing in the output said so.
#
# Arms are named for what they test, and the name is also the run directory --
# so it is what every log and every table cites. The verdicts are in
# `docs/RESULTS.md`; what is here is the configuration behind them, and every
# arm trains on the **synthetic** source because that is where the table is
# measured.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

CONFIG="${MPF_CONFIG:-configs/synth_base.yaml}"

# **COMMON prefixes every arm, so a field added here silently redefines every
# arm that has already run.** Nothing in an arm line shows it, and two arms
# launched a week apart then differ by whatever moved in between. Change it
# only by re-running the cells that are read against each other.
#
# Three loader workers each, not the default eight: these runs are loader-bound
# on a 20-core host, so oversubscribing the CPU starves the GPU rather than
# sharing it. Six arms at three workers is the measured optimum --
#
#   6 arms x 3 workers   63.5 fps each   381 fps aggregate
#   8 arms x 2 workers   39.9 fps each   320 fps aggregate
#
# -- so when the host is full, wait for a slot: a queued arm finishes sooner
# than one that slows every other arm down. The GPU barely matters; the
# element-token arm ran on a nominally 3x slower card at 64.0 fps against
# 62.5-64.2 on the large one. Use `MPF_GPU_INDEX` to spread arms, not to speed
# them up.
#
# **Both geometry fields are pinned, and that is not redundancy.** The model
# reads `geometry` when tokens are elements and `point_geometry` when they are
# points, so an arm setting neither gets a different score geometry depending
# on the very knob under test -- which is how point+rope gets compared against
# element+relative and called a tokens experiment.
#
# `data.cache_dir` reads a pre-generated split rather than rebuilding every
# sample each epoch. Exact, not approximate: `set_epoch` is never called, so
# every epoch already sees identical data. The loader goes from 335 to 2732
# samples/s, 8.2x, contents verified identical. Set MPF_CACHE= to disable.
#
# 25 epochs as a ceiling with early stopping deciding the real number, and
# patience 12 because the longest plateau an arm sat on *before* reaching its
# eventual best was 8. The ceiling is not universal: four-layer point-token
# arms peak at epoch 18-19, two-layer element arms at 29-31. Queue one of the
# latter and raise it, because truncation is the dangerous half -- an arm
# stopped before its peak looks worse than one that reached its own, and the
# comparison then measures the schedule instead of the structure.
#
# MPF_RESUME=1 continues an arm from its own last.pt instead of starting over.
#
# **Stopping an arm does not mean deleting it.** Kill its process group, move
# its `*.log` aside so the launcher below stops refusing the name, and relaunch
# with MPF_RESUME=1; the arm picks up from the epoch it reached. Removing the
# run directory instead once cost 51 recoverable GPU-hours, and disk has never
# been the constraint here. A `for d in runs/glob*/` loop that deletes
# `runs/$(basename $d)` becomes `rm -rf runs/glob*` the moment the glob matches
# nothing, so guard it with `[ -d "$d" ]`.
# The scheduler sets it when retrying a crashed or stalled arm; a fresh launch
# never does, because silently resuming is how you train a model whose history
# you cannot account for.
RESUME_FLAG=""
[ -n "${MPF_RESUME:-}" ] && RESUME_FLAG="train.resume=true"

COMMON="${RESUME_FLAG} model.layers=4 train.epochs=${MPF_EPOCHS:-25} train.patience=${MPF_PATIENCE:-12} train.num_workers=${MPF_WORKERS:-3} model.geometry=rope model.point_geometry=rope data.cache_dir=${MPF_CACHE-cache}"

# name                     overrides
#
# The tokens, residual and augmentation arms pin `grad_clip` at 1.0; everything
# below takes the default of 10.0. Stated on the arm line rather than inherited
# from COMMON, because arms either side of that boundary are not comparable and
# the line is the only place a reader looks.
ARMS=(
  # --- tokens and residual -------------------------------------------------
  "tokens_point_s0|model.tokens=point   model.residual=line  train.seed=0 train.grad_clip=1.0"
  "tokens_point_s1|model.tokens=point   model.residual=line  train.seed=1 train.grad_clip=1.0"
  "tokens_point_s2|model.tokens=point   model.residual=line  train.seed=2 train.grad_clip=1.0"

  # Three seeds, because the residual reversal rests on seed 0 alone -- 98.0%
  # recall against point-to-line's 94.4/96.6/96.6, at 1.7 sd -- and calibration
  # is noisier still: those three line seeds span a factor of five in NEES, so
  # one value outside their range settles nothing.
  "residual_point_s0|model.tokens=point   model.residual=point train.seed=0 train.grad_clip=1.0"
  "residual_point_s1|model.tokens=point   model.residual=point train.seed=1 train.grad_clip=1.0"
  "residual_point_s2|model.tokens=point   model.residual=point train.seed=2 train.grad_clip=1.0"

  # Element tokens under `relative` -- the geometry that path was designed
  # around, so it answers a different question from the control below.
  "tokens_element_relative_s0|model.tokens=element model.residual=line model.geometry=relative train.seed=0 train.grad_clip=1.0"
  # The single-variable control: element tokens under the score geometry the
  # point arms run. Three seeds, because a control with n=1 against a treatment
  # with n=3 puts the whole standard error on one side -- at the measured seed
  # sd of 0.0089 m, SE(diff) is 0.0103 for 3-vs-1 and 0.0073 for 3-vs-3.
  "tokens_element_s0|model.tokens=element model.residual=line  train.seed=0 train.grad_clip=1.0"
  "tokens_element_s1|model.tokens=element model.residual=line  train.seed=1 train.grad_clip=1.0"
  "tokens_element_s2|model.tokens=element model.residual=line  train.seed=2 train.grad_clip=1.0"

  # --- heads: count at fixed width, width at fixed count -------------------
  #
  # 2x64 is the control at 1.709 M parameters, 4x64 doubles the attention width
  # at the same rank for 2.764 M, 2x32 halves the rank cap at the same count
  # for 1.181 M.
  #
  # **`rope_bands` is restated on all three because `head_dim` moves**: it
  # derives from `head_dim // 6`, so 2x32 would take 5 bands against the
  # others' 10 and the contrast would sweep positional bandwidth as well as
  # rank. 5 is the most `head_dim` 32 can carry, so 5 is what all three take.
  "heads_2x64_s0|model.tokens=point model.residual=line model.heads=2 model.head_dim=64 model.rope_bands=5 train.seed=0"
  "heads_2x64_s1|model.tokens=point model.residual=line model.heads=2 model.head_dim=64 model.rope_bands=5 train.seed=1"
  "heads_4x64_s0|model.tokens=point model.residual=line model.heads=4 model.head_dim=64 model.rope_bands=5 train.seed=0"
  "heads_4x64_s1|model.tokens=point model.residual=line model.heads=4 model.head_dim=64 model.rope_bands=5 train.seed=1"
  "heads_2x32_s0|model.tokens=point model.residual=line model.heads=2 model.head_dim=32 model.rope_bands=5 train.seed=0"
  "heads_2x32_s1|model.tokens=point model.residual=line model.heads=2 model.head_dim=32 model.rope_bands=5 train.seed=1"

  # --- depth ---------------------------------------------------------------
  #
  # `layers` is the one structural knob COMMON fixes rather than measures, so
  # both cells are launched here together. A control inherited from an earlier
  # launch differs by whatever COMMON has changed since, and nothing in the arm
  # lines would show it.
  "layers2_s0|model.tokens=point model.residual=line model.layers=2 train.seed=0"
  "layers2_s1|model.tokens=point model.residual=line model.layers=2 train.seed=1"
  "layers2_s2|model.tokens=point model.residual=line model.layers=2 train.seed=2"
  "layers4_s0|model.tokens=point model.residual=line model.layers=4 train.seed=0"
  "layers4_s1|model.tokens=point model.residual=line model.layers=4 train.seed=1"
  "layers4_s2|model.tokens=point model.residual=line model.layers=4 train.seed=2"

  # --- geometry: does the encoding in the score do any work? ---------------
  #
  # `absolute` adds position to the token and learns whatever invariance it
  # can; `rope` carries relative position into the score and is
  # translation-invariant by construction. Two points on parallel lane lines at
  # the same station have identical content features, so the only thing
  # separating them is their frames -- if `absolute` collapses, the
  # equivariance claim is load-bearing. `geom_rope_*` is the control, launched
  # with the treatment, and is the reference for anything added here later.
  "geom_rope_s0|model.tokens=point model.residual=line train.seed=0"
  "geom_rope_s1|model.tokens=point model.residual=line train.seed=1"
  "geom_rope_s2|model.tokens=point model.residual=line train.seed=2"
  "geom_abs_s0|model.tokens=point model.residual=line model.point_geometry=absolute train.seed=0"
  "geom_abs_s1|model.tokens=point model.residual=line model.point_geometry=absolute train.seed=1"
  "geom_abs_s2|model.tokens=point model.residual=line model.point_geometry=absolute train.seed=2"

  # --- width ---------------------------------------------------------------
  #
  # **`rope_bands` again on every cell, including the 128-wide one**, because
  # `dim` moves `head_dim` and `head_dim` moves the derived band count: 5 at
  # dim 64, 10 at dim 128, 21 at dim 256. Unpinned, this sweeps positional
  # bandwidth alongside width. `geom_rope_*` is also 128 wide but lets the
  # count derive, so it is not the control for width.
  #
  # 21 bands is not merely more, it is broken: the frequencies are
  # `(2*pi/30) * 2^k`, so band 21 reaches 1.1e7 radians of phase at a 50 m
  # coordinate and the identity rope exists for -- a score depending only on
  # the relative offset -- fails in fp32 by 15% mean and 28% worst under
  # translation.
  #
  # `dim` 256 needs 6565 MiB forward and cannot finish a backward pass in
  # 7.6 GiB, so those cells need a large card. `dim` 64 gets a third seed,
  # because the pinned-band cells are predicted noisy and two seeds cannot show
  # that.
  "dim64_s0|model.tokens=point model.residual=line model.dim=64 model.rope_bands=5 train.seed=0"
  "dim64_s1|model.tokens=point model.residual=line model.dim=64 model.rope_bands=5 train.seed=1"
  "dim64_s2|model.tokens=point model.residual=line model.dim=64 model.rope_bands=5 train.seed=2"
  "dim128_s0|model.tokens=point model.residual=line model.dim=128 model.rope_bands=5 train.seed=0"
  "dim128_s1|model.tokens=point model.residual=line model.dim=128 model.rope_bands=5 train.seed=1"
  "dim256_s0|model.tokens=point model.residual=line model.dim=256 model.rope_bands=5 train.seed=0"
  "dim256_s1|model.tokens=point model.residual=line model.dim=256 model.rope_bands=5 train.seed=1"

  # --- is 5 bands the optimum, or merely better than 10? -------------------
  #
  #   bands  3   18 rotated, 46 content, shortest wavelength 7.500 m
  #   bands  5   30 rotated, 34 content, shortest wavelength 1.875 m
  #   bands 10   60 rotated,  4 content, shortest wavelength 0.059 m
  #
  # 5 beats 10 at `head_dim` 64, but those two points bracket one side of 5
  # only. Three is past where the wavelength argument allows -- 7.5 m is four
  # times the map point pitch -- so if content capacity still wins there, the
  # teacher should go below 5; if it loses, 5 is near the optimum.
  "bands3_s0|model.tokens=point model.residual=line model.dim=128 model.rope_bands=3 train.seed=0"

  # One override against `dim256_s0`: 10 bands instead of 5. Every width cell
  # pins 5, whose shortest wavelength is 1.875 m against a 1.714 m point pitch,
  # so the model cannot separate adjacent map points -- and a ceiling imposed
  # by the encoding looks exactly like saturation in width. `residual=line` to
  # match `dim256_s0` exactly; changing two things makes it unreadable.
  "dim256_bands10_s0|model.tokens=point model.residual=line model.dim=256 model.rope_bands=10 train.seed=0"

  # --- does augmentation remove the overfit? -------------------------------
  #
  # 40 epochs rather than 25, because the question *is* the shape of the curve
  # past epoch 19, and no cache, because a materialised split cannot redraw its
  # noise -- that is the mechanism of augmentation here, and of its absence.
  #
  # `noaug_*` exists because **the config field does not distinguish the two
  # conditions**: `augment: true` is recorded whether or not `set_epoch` is
  # ever called, so nothing on disk says which arms actually saw fresh noise.
  # `data.augment=false` is stated rather than left implicit so the checkpoint
  # records the condition honestly.
  "aug_on_s0|model.tokens=point model.residual=line train.seed=0 train.grad_clip=1.0 train.epochs=40 data.cache_dir="
  "aug_on_s1|model.tokens=point model.residual=line train.seed=1 train.grad_clip=1.0 train.epochs=40 data.cache_dir="
  "aug_on_s2|model.tokens=point model.residual=line train.seed=2 train.grad_clip=1.0 train.epochs=40 data.cache_dir="
  "aug_off_s0|model.tokens=point model.residual=line train.seed=0 train.grad_clip=1.0 train.epochs=40 data.augment=false"
  "aug_off_s1|model.tokens=point model.residual=line train.seed=1 train.grad_clip=1.0 train.epochs=40 data.augment=false"
  "aug_off_s2|model.tokens=point model.residual=line train.seed=2 train.grad_clip=1.0 train.epochs=40 data.augment=false"

  # --- the teacher ---------------------------------------------------------
  #
  # Everything the sweeps above settled, with the margins in `docs/RESULTS.md`:
  # point tokens, the point-to-point residual, four layers, two heads, and
  # `dim` 128, where accuracy saturates. `dim` 128 and `heads` 2 are defaults
  # and so are not restated; that makes `head_dim` derive to 64, which is what
  # `inner = dim` requires.
  #
  # **`rope_bands` must be restated a third time**, for the same reason: 5, not
  # the 10 that `head_dim` 64 derives. `RotaryFrames` rotates `6 * bands`
  # channels and passes the rest through, so 10 would leave *four* channels for
  # anything that is not position, and 5 bands beat 10 by 3.05 pp at 2.5 sd.
  # The wavelength argument for 10 was half the story: 5 bands reach only
  # 1.875 m against a 1.714 m point pitch, which costs less than having almost
  # no content channels.
  "teacher|model.tokens=point model.residual=point model.rope_bands=5 train.seed=0 train.epochs=40"

  # --- the student, and the control that makes it readable -----------------
  #
  # **Shallow, not narrow.** Width saturates at 128, so narrowing costs
  # accuracy and buys nothing, while `tools/cost.py` puts this model
  # launch-bound at batch 1: 16x the parameters moved latency 2%, and halving
  # *depth* moved it 29%. Depth is the only compression this architecture
  # rewards, and closing the 9.8 pp depth gap is what the teacher is for.
  #
  # `student_ctl_*` is the same arm with no teacher. Without it, "distillation
  # recovered X" is not a measurement. Three seeds a side, not two: the first
  # two controls came in 5.8 pp apart, wider than any distillation effect this
  # project could claim, and two seeds a side cannot separate anything under
  # about 8 pp.
  "student_s0|model.tokens=point model.residual=point model.layers=2 model.rope_bands=5 train.seed=0 train.epochs=40 distill.teacher=runs/teacher/best.pt"
  "student_s1|model.tokens=point model.residual=point model.layers=2 model.rope_bands=5 train.seed=1 train.epochs=40 distill.teacher=runs/teacher/best.pt"
  "student_s2|model.tokens=point model.residual=point model.layers=2 model.rope_bands=5 train.seed=2 train.epochs=40 distill.teacher=runs/teacher/best.pt"
  "student_no_teacher_s0|model.tokens=point model.residual=point model.layers=2 model.rope_bands=5 train.seed=0 train.epochs=40"
  "student_no_teacher_s1|model.tokens=point model.residual=point model.layers=2 model.rope_bands=5 train.seed=1 train.epochs=40"
  "student_no_teacher_s2|model.tokens=point model.residual=point model.layers=2 model.rope_bands=5 train.seed=2 train.epochs=40"

  # --- compressing the teacher ---------------------------------------------
  #
  # `tools/prune.py` removes only the FFN hidden width -- eight modules -- and
  # at `dim` 128 the FFN is a minority of the parameters: keeping 0.75 removes
  # 7.7% of the model and keeping 0.50 removes 15.4%, against the 25% and 50%
  # the knob's name suggests. That is the honest ceiling of structural pruning
  # here, which is why the question is what the fine-tune recovers rather than
  # how small the result gets -- and it must be read on calibration too, since
  # pruning costs more NEES than accuracy. Ten epochs, not forty: recovery from
  # a perturbation, not training from scratch.
  "prune_keep75_finetuned|model.tokens=point model.residual=point model.rope_bands=5 train.seed=0 train.init_from=runs/prune_keep75/init.pt train.epochs=10"
  "prune_keep50_finetuned|model.tokens=point model.residual=point model.rope_bands=5 train.seed=0 train.init_from=runs/prune_keep50/init.pt train.epochs=10"

  # The control those two need. Both come back BETTER than the teacher, the
  # more aggressive prune winning by more, and pruning does not do that. What
  # they also share is ten further epochs from the teacher's `best.pt` under a
  # **fresh warmup-and-cosine schedule**, and the teacher's best epoch was 19
  # of a run that early-stopped at 31 -- twelve epochs of no improvement, which
  # a learning-rate restart is a known way off. So "pruned" is confounded with
  # "trained longer on a restarted schedule" and nothing on disk separates
  # them. This arm is the same ten epochs with no pruning: whatever it gains is
  # the restart, and only what the pruned arms gain *beyond* it is pruning.
  "teacher_finetuned|model.tokens=point model.residual=point model.rope_bands=5 train.seed=0 train.init_from=runs/teacher/best.pt train.epochs=10"
)

# What one arm needs on the card. `tools/cost.py` measures a 128-wide
# point-token arm at 5006 MiB peak at batch 64 and `dim` 256 at 6565 MiB
# *forward only*, whose backward pass will not fit 8 GiB at all -- so one flat
# constant would admit `dim` 256 to a card that provably cannot run it, and it
# dies minutes in, after the launcher has reported success.
#
# 8000 is measured, not guessed: a default-width arm was admitted to a card
# with 7600 MiB free and died at the assignment softmax holding 7.09 GiB, 108
# MiB short -- twice, the second time with expandable segments on, so it is
# capacity and not fragmentation. `dim64` is separate because it does fit, and
# runs on an 8 GiB card at 97% utilisation.
case "${1:-}" in
  *dim256*) NEED_MIB="${MPF_NEED_MIB:-12000}" ;;
  *dim64*)  NEED_MIB="${MPF_NEED_MIB:-7000}"  ;;
  *)        NEED_MIB="${MPF_NEED_MIB:-8000}"  ;;
esac
gpu="${MPF_GPU_INDEX:-0}"

check_vram () {
  # Free memory alone is not enough, because an arm takes minutes to reach its
  # peak: two launched back to back each see the card empty, both start, and
  # both die at their first step. So count what is already *starting* here and
  # reserve NEED_MIB for each.
  #
  # Only our trainers count. A host running a desktop also lists its terminal,
  # system monitor and GPU control panel under `--query-compute-apps`, three
  # processes holding 127 MiB between them; counting those as arms still
  # growing made the guard demand 24 GiB on an 8 GiB card and refuse it
  # permanently.
  #
  # **Growing means young, not small.** A memory threshold is a proxy for youth
  # and a bad one -- a `dim64` arm's *steady state* is 3286 MiB, so every small
  # arm would count as growing for its whole life. An arm reaches its peak
  # within about three minutes of its first step and launches are spaced by
  # 45 s, so four minutes is the window with margin.
  local free want growing=0 age
  free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  [ -z "$free" ] && return 0
  for pid in $(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    ps -p "$pid" -o args= 2>/dev/null | grep -q "tools/train.py" || continue
    age=$(ps -p "$pid" -o etimes= 2>/dev/null | tr -d ' ')
    [ -n "$age" ] && [ "$age" -lt "${MPF_GROW_S:-240}" ] && growing=$((growing + 1))
  done
  want=$(( NEED_MIB + growing * NEED_MIB ))
  if [ "$free" -lt "$want" ]; then
    echo "REFUSING: GPU $gpu has ${free} MiB free; this arm needs ${NEED_MIB}" >&2
    echo "  and ${growing} arm(s) there are still growing, so ${want} is required." >&2
    echo "  Use MPF_GPU_INDEX for another card, wait for a slot, or set" >&2
    echo "  MPF_NEED_MIB if you know this arm is smaller." >&2
    return 1
  fi
  return 0
}

check_vram || exit 1

# A configured cache that is not there is not a slow start, it is a dead arm:
# CachedDataset raises at construction and the run ends at step zero, minutes
# after the launcher reported success. The cache takes 45 minutes to build and
# exists only on the training host, so the gap is a real state to refuse.
want_cache="${MPF_CACHE-cache}"
if [ -n "$want_cache" ]; then
  for split in train val; do
    if [ ! -f "$want_cache/$split.pt" ]; then
      echo "REFUSING: data.cache_dir=$want_cache but $want_cache/$split.pt is missing." >&2
      echo "  Build it with tools/build_cache.py, or set MPF_CACHE= to generate" >&2
      echo "  samples on the fly." >&2
      exit 1
    fi
  done
fi

mkdir -p runs
want="${1:-}"
launched=0

for spec in "${ARMS[@]}"; do
  name="${spec%%|*}"
  overrides="${spec#*|}"
  [ -n "$want" ] && [ "$want" != "$name" ] && continue

  # **A student must wait for its teacher to FINISH, not merely to exist.**
  # `best.pt` is rewritten every epoch a teacher improves, so it appears within
  # minutes of the teacher starting: a `-f "$teacher_ckpt"` guard passes and
  # the student distils from a teacher ten epochs into forty. Nothing would say
  # so -- the arm trains, converges, and reports a number that gets read as
  # what distillation is worth. The launcher's DONE line is the only durable
  # statement that a run completed, so that is what this waits for.
  #
  # Inside the loop, because `overrides` exists only here: the same check one
  # level up reads an unset variable, and `set -u` then kills the guard instead
  # of the launch.
  teacher_ckpt=$(printf '%s\n' "$overrides" \
    | grep -oE 'distill\.teacher=[^ ]+' | sed 's/.*=//')
  if [ -n "${teacher_ckpt:-}" ]; then
    teacher_arm=$(basename "$(dirname "$teacher_ckpt")")
    if ! grep -qh "^########## DONE ${teacher_arm} rc=0" \
         runs/"${teacher_arm}"_2*.log 2>/dev/null; then
      echo "REFUSING $name: teacher ${teacher_arm} has not finished." >&2
      echo "  Its best.pt may already exist -- it is rewritten every epoch --" >&2
      echo "  but distilling from a partly trained teacher measures nothing." >&2
      continue
    fi
    [ -f "$teacher_ckpt" ] || { echo "REFUSING $name: $teacher_ckpt missing" >&2; continue; }
  fi

  script="runs/.launch_${name}.sh"
  # Written with an unexpanded heredoc and then given its values by argument,
  # so nothing here depends on what was or was not set in a parent shell.
  cat > "$script" <<'INNER'
#!/bin/bash
set -uo pipefail
cd "$1" || exit 1
. scripts/env.sh && mpf_activate 2>/dev/null || true
echo "########## $2"
echo "########## $3 $4"
# An 8 GiB card holds a 4-layer point-token arm only if the allocator does not
# waste the difference: one died asking for 108 MiB while holding 251 MiB
# reserved-but-unallocated, which is fragmentation rather than capacity.
# Expandable segments give that back, and are harmless on a larger card.
#
# Batch size is deliberately left alone. Shrinking it would fit trivially and
# would also make the arm incomparable to its own siblings -- a structural
# verdict read off arms that differ in their optimiser as well as their
# structure is not a verdict.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${MPF_GPU_INDEX:-0}" python3 tools/train.py \
  --config "$3" $4 "train.out_dir=runs/$2"
rc=$?
echo "########## DONE $2 rc=$rc"
INNER
  chmod +x "$script"

  log="runs/${name}_$(date +%Y%m%d-%H%M%S).log"
  nohup setsid bash "$script" "$PWD" "$name" "$CONFIG" \
    "$COMMON $overrides" > "$log" 2>&1 < /dev/null &
  echo "launched $name -> $log"
  launched=$((launched + 1))
  # Long enough for the arm to claim its memory, so the next check sees it.
  sleep "${MPF_LAUNCH_GAP:-45}"
done

[ "$launched" -eq 0 ] && echo "no arm matched ${want:-<all>}" && exit 1
exit 0
