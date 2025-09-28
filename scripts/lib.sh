#!/bin/bash

# lib.sh -- shared by the dataset download scripts.
#
# Sourced by download_nuscenes.sh and download_argoverse2.sh, and nothing else.
# Two callers would be thin justification for a shared file; 200 lines of
# resumable parallel-range fetching, identical for both, is not.

#: Where every dataset lands. Absolute, on the data drive, and deliberately
#: outside the checkout: scripts/remote-ubuntu.sh --sync runs rsync --delete
#: over the checkout and would take a dataset with it.
data_root() {
  echo "${MPF_DATA_ROOT:-/DATA/map-pose-former}"
}

#: Pieces fetched at once, per file. Sixteen is measured, not chosen: from this
#: host one connection to S3 sustains 17 kB/s and sixteen sustain 408 kB/s --
#: 4 hours against 104 for a 6.4 GB tar. The limit is per-connection throttling
#: somewhere on the path rather than bandwidth, which is why nuScenes over
#: CloudFront saturates the line on a single stream and Argoverse over S3 does
#: not. Thirty-two measured *slower* than sixteen, so the answer is not "more".
CONNECTIONS="${MPF_CONNECTIONS:-16}"

#: Set by --force. Resuming is the default because these are multi-hour
#: downloads on an unreliable path and re-running is the normal way to finish
#: one; discarding good bytes has to be asked for explicitly.
FORCE="${MPF_FORCE:-0}"

# usage FILE -- print FILE's header comment block, without the markers.
#
# Bounded by the first non-comment line rather than by a line number. Every
# hardcoded `sed -n 'A,Bp'` range in this repo had already drifted past the end
# of its own header and was printing `set -euo pipefail` as though it were
# prose -- a range is a second copy of where the header ends, and the two
# copies disagree the moment a line is added.
usage() {
  # `started` matters: line 2 is the blank between the shebang and the header,
  # and exiting on the first non-comment line without it would print nothing.
  awk 'NR == 1 { next }
       /^#/    { sub(/^# ?/, ""); print; started = 1; next }
       started { exit }' "$1"
}

# _head URL -- final status and content-length of a redirect chain, as "code size".
_head() {
  local h
  h=$(curl -sSIL --max-time 30 "$1" | tr -d '\r') || return 1
  # The last of each, because these hosts redirect and awk keeps overwriting:
  # what survives to END is the end of the chain.
  awk '/^HTTP\//{c=$2} tolower($1)~/^content-length:/{n=$2} END{print c, n}' <<< "${h}"
}

# _fetch_range URL PART START END -- one piece, retried until it is whole.
_fetch_range() {
  local url="$1" part="$2" start="$3" end="$4"
  local want=$((end - start + 1)) have attempt
  for attempt in $(seq 1 10); do
    have=$(stat -c%s "${part}" 2>/dev/null || echo 0)
    [ "${have}" -ge "${want}" ] && return 0
    # Resume by reissuing the range from where this piece stopped, rather than
    # with `curl -C -`, which resumes against the file's own offset and means
    # something different once a range is in play.
    #
    # --speed-limit/--speed-time is the flag that matters overnight: a stalled
    # socket that never closes is the failure mode, and without this curl waits
    # on it forever instead of retrying.
    # -f so an error body is not appended. Without it curl writes the server's
    # 503 XML into the middle of a binary piece, cheerfully, and reports
    # success by exit code alone.
    curl -sSf --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
      -r "$((start + have))-${end}" "${url}" >> "${part}" 2>/dev/null || true

    # A piece cannot legitimately exceed its own range, so if it has, something
    # else wrote into it -- a second copy of this script, or a server that
    # ignored the Range and sent the whole body. Either way the bytes are
    # interleaved and unusable, and keeping them would produce a file of
    # plausible length and wrong content. Start the piece over.
    if [ "$(stat -c%s "${part}" 2>/dev/null || echo 0)" -gt "${want}" ]; then
      echo "  ${part##*/}: overran its range, discarding and refetching" >&2
      : > "${part}"
    fi
    sleep 5
  done
  have=$(stat -c%s "${part}" 2>/dev/null || echo 0)
  [ "${have}" -ge "${want}" ]
}

# _fetch_file URL DEST -- one file, in CONNECTIONS pieces, resumable.
_fetch_file() {
  local url="$1" dest="$2" size="$3" ranged="$4"
  # Separate statements: `local a=1 b=$a` marks both names local before it
  # assigns either, so $a is still unset when b is evaluated -- which under
  # `set -u` is an error rather than an empty string.
  local name="${url##*/}"
  local out="${dest}/${name}" parts="${dest}/.parts"
  local chunk i s e pids=() rc=0

  if [ "${FORCE}" = 1 ]; then
    echo "  ${name}: --force, discarding what is there"
    rm -f "${out}" "${parts}/${name}.layout"
    for i in $(seq 0 200); do rm -f "${parts}/${name}.${i}"; done
  fi

  if [ -f "${out}" ] && [ "$(stat -c%s "${out}")" = "${size}" ]; then
    echo "${name}: complete already"
    return 0
  fi

  # A server that ignores Range answers 200 with the whole body, and splitting
  # against it would write CONNECTIONS copies of the file into the parts and
  # concatenate them into garbage. Check, rather than assume.
  if [ "${ranged}" != "yes" ]; then
    echo "${name}: no range support, single stream"
    curl -sS --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
      -C - -o "${out}" "${url}" || return 1
    return 0
  fi

  mkdir -p "${parts}"
  chunk=$(((size + CONNECTIONS - 1) / CONNECTIONS))

  # Pieces are only resumable against the split that produced them. Change
  # MPF_CONNECTIONS between runs and piece 3 now starts somewhere piece 3 has
  # never been, so appending to it would interleave two different offsets into
  # one file -- a corruption the length check cannot see, because the total is
  # still right. Record the layout and start over if it moved.
  local stamp="${parts}/${name}.layout"
  if [ -f "${stamp}" ] && [ "$(cat "${stamp}")" != "${CONNECTIONS}:${size}" ]; then
    echo "  ${name}: split changed since the last run, discarding pieces" >&2
    for i in $(seq 0 200); do rm -f "${parts}/${name}.${i}"; done
  fi
  echo "${CONNECTIONS}:${size}" > "${stamp}"
  for i in $(seq 0 $((CONNECTIONS - 1))); do
    s=$((i * chunk))
    e=$((s + chunk - 1))
    [ "${e}" -ge "${size}" ] && e=$((size - 1))
    [ "${s}" -gt "${e}" ] && continue
    _fetch_range "${url}" "${parts}/${name}.${i}" "${s}" "${e}" &
    pids+=($!)
  done
  # Progress is a size poll rather than curl's own meter: sixteen meters
  # interleaved on one terminal is unreadable, and one number is what a person
  # checking on a run overnight actually wants.
  # Redrawn in place on a terminal, appended on a schedule when it is a log --
  # a carriage return does not erase anything in a file, and these runs are
  # detached for four hours, so \r would write a single unreadable line.
  local tick=20 nl='\r'
  if [ ! -t 1 ]; then tick=300; nl='\n'; fi
  ( while kill -0 "${pids[0]}" 2>/dev/null; do
      printf "  %s  %6.2f / %.2f GB${nl}" "${name}" \
        "$(bc -l <<< "$(cat "${parts}/${name}".* 2>/dev/null | wc -c)/1000000000")" \
        "$(bc -l <<< "${size}/1000000000")"
      sleep "${tick}"
    done ) &
  local meter=$!
  for p in "${pids[@]}"; do wait "${p}" || rc=1; done
  kill "${meter}" 2>/dev/null || true
  printf '\r'
  [ "${rc}" -eq 0 ] || { echo "${name}: incomplete" >&2; return 1; }

  # By index, never by glob. `cat name.*` orders lexicographically -- .0 .1 .10
  # .11 ... .2 -- so sixteen pieces concatenate in the wrong order into a file
  # of exactly the right length. Nothing downstream would catch it until a tar
  # failed to extract hours later, and the size check here certainly would not.
  : > "${out}"
  for i in $(seq 0 $((CONNECTIONS - 1))); do
    [ -f "${parts}/${name}.${i}" ] && cat "${parts}/${name}.${i}" >> "${out}"
  done
  if [ "$(stat -c%s "${out}")" != "${size}" ]; then
    echo "${name}: assembled size is wrong; leaving pieces in place" >&2
    rm -f "${out}"
    return 1
  fi
  for i in $(seq 0 $((CONNECTIONS - 1))); do rm -f "${parts}/${name}.${i}"; done
  rm -f "${stamp}"
  echo "  ${name}: done"
}

# download_set DEST URL...
#
# Sum the transfer first, refuse it if it will not fit, then fetch each file in
# parallel pieces. Asking before starting is the point: these run overnight,
# and one that fills the disk at 3 a.m. leaves a half-extracted dataset behind
# a message nobody is awake to read. Twice the archive size, because the
# archives are kept -- a re-run should not refetch 50 GB to extract it again.
download_set() {
  local dest="$1"; shift
  local urls=("$@") total=0 free code size ranged
  local sizes=() ranges=()

  mkdir -p "${dest}"

  # One writer per destination. Two copies of this script append to the same
  # pieces from different offsets, and the result is a file of roughly the
  # right size whose contents are two transfers shuffled together -- which is
  # exactly what happened the first time this ran, when a stopped instance left
  # its curls behind and the restart joined them.
  #
  # flock -n rather than waiting: a second run is a mistake to report, not a
  # queue to join. The fd is held for the life of the shell, so a kill -9
  # releases it and the next run proceeds.
  exec 9>"${dest}/.lock"
  if ! flock -n 9; then
    echo "another download is already running in ${dest}" >&2
    echo "wait for it, or stop it -- including its curl children:" >&2
    echo "  pkill -f 'download_.*[.]sh'; pkill -f 'curl.*${dest##*/}'" >&2
    return 1
  fi
  for u in "${urls[@]}"; do
    # The status has to be checked, not only the length: an error page is an
    # ordinary response with an ordinary Content-Length, so a preflight that
    # only asks "how big" sums a directory of 404s and hands wget the surprise.
    read -r code size <<< "$(_head "${u}")"
    case "${code}" in
      2*) ;;
      "") echo "no answer from ${u}" >&2; return 1 ;;
      *)  echo "HTTP ${code} for ${u}" >&2; return 1 ;;
    esac
    [ -n "${size}" ] || { echo "no content-length for ${u}" >&2; return 1; }
    ranged=no
    [ "$(curl -sSI -r 0-0 --max-time 30 "${u}" | awk '/^HTTP\//{c=$2} END{print c}' | tr -d '\r')" = "206" ] && ranged=yes
    sizes+=("${size}"); ranges+=("${ranged}")
    total=$((total + size))
  done

  free=$(df -B1 --output=avail "${dest}" | tail -1)
  printf '%d files, %.1f GB, into %s\n' "${#urls[@]}" "$(bc -l <<< "${total}/1000000000")" "${dest}"
  printf 'free space: %.1f GB, %s connections per file\n\n' \
    "$(bc -l <<< "${free}/1000000000")" "${CONNECTIONS}"
  if [ "${free}" -lt $((total * 2)) ]; then
    echo "not enough room: extraction needs about the archive size again" >&2
    return 1
  fi

  local failed=() i=0
  for u in "${urls[@]}"; do
    echo "=== ${u##*/}"
    _fetch_file "${u}" "${dest}" "${sizes[$i]}" "${ranges[$i]}" || failed+=("${u##*/}")
    i=$((i + 1))
  done

  # Carry on through a failure rather than aborting on it. One unreachable file
  # at 2 a.m. should not cost the other nine their night, and every one of them
  # resumes on the next run anyway.
  if [ ${#failed[@]} -gt 0 ]; then
    echo >&2
    echo "${#failed[@]} of ${#urls[@]} files failed: ${failed[*]}" >&2
    echo "re-run to resume -- finished files are skipped, partial ones continue." >&2
    return 1
  fi
}
