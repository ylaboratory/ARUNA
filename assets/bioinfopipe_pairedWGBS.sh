#!/usr/bin/env bash

# ------------------------------------------------------------
# NOTE ON SINGLE-END / RRBS DATA
#
# This script assumes *paired-end WGBS* input and uses:
#   - trim_galore --paired
#   - bismark paired-end alignment
#
# Users working with *single-ended RRBS* data should instead use
# the following canonical commands (shown for reference):
#
#   trim_galore --fastqc --gzip --cores <N> --rrbs sample.fastq.gz \
#       --output_dir <OUT_DIR>
#
#   bismark --genome <GENOME_DIR> --se sample_trimmed.fq.gz \
#       --fastq --parallel <N> --gzip --output_dir <OUT_DIR>
# 
# For RRBS, deduplication is skipped.
# For single-end sequenced WGBS, bismark is run in --se mode same as above.
# Downstream steps (methylation extraction, CpG merging) remain unchanged.
#
# ------------------------------------------------------------

set -euo pipefail

usage() {
  cat <<'EOF'
Single-sample WGBS/RRBS processing (Trim Galore -> Bismark -> dedup -> methylation extractor -> merge CpG).

Required:
  -s, --sample-id       Sample ID (used for output subdir)
  --r1                  Path to R1 FASTQ(.gz)
  --r2                  Path to R2 FASTQ(.gz)
  -g, --genome          Bismark genome folder (contains Bisulfite_Genome/)

Optional:
  -o, --out-root        Output root directory (default: current directory)
  --trim-cores          Cores for Trim Galore (default: 8)
  --bismark-cores       Cores for Bismark + extractor (default: 12)
  --link-inputs         Symlink FASTQs into output dir instead of copying
  --keep-intermediates  Do NOT delete *.bam, *.fq.gz, *.txt.gz intermediates
  -h, --help            Show help

Example:
  ./run_one_sample.sh -s EGAN0000 --r1 /data/A_R1.fastq.gz --r2 /data/A_R2.fastq.gz \
    -g /path/to/hg38 --trim-cores 16 --bismark-cores 24 -o ./runs --keep-intermediates
EOF
}

# -----------------------------
# Defaults (can be overridden)
# -----------------------------
OUT_ROOT="$(pwd)"
TRIMGALORE_CORES=8
BISMARK_CORES=12
LINK_INPUTS=0
KEEP_INTERMEDIATES=0

SAMPLE_ID=""
R1=""
R2=""
BISMARK_GENOME=""

# -----------------------------
# Parse args
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--sample-id) SAMPLE_ID="$2"; shift 2;;
    --r1) R1="$2"; shift 2;;
    --r2) R2="$2"; shift 2;;
    -g|--genome) BISMARK_GENOME="$2"; shift 2;;
    -o|--out-root) OUT_ROOT="$2"; shift 2;;
    --trim-cores) TRIMGALORE_CORES="$2"; shift 2;;
    --bismark-cores) BISMARK_CORES="$2"; shift 2;;
    --link-inputs) LINK_INPUTS=1; shift 1;;
    --keep-intermediates) KEEP_INTERMEDIATES=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown argument: $1"; usage; exit 1;;
  esac
done

# -----------------------------
# Validate
# -----------------------------
if [[ -z "${SAMPLE_ID}" || -z "${R1}" || -z "${R2}" || -z "${BISMARK_GENOME}" ]]; then
  echo "[ERROR] Missing required arguments."
  usage
  exit 1
fi

if [[ ! -f "${R1}" ]]; then echo "[ERROR] R1 not found: ${R1}"; exit 1; fi
if [[ ! -f "${R2}" ]]; then echo "[ERROR] R2 not found: ${R2}"; exit 1; fi
if [[ ! -d "${BISMARK_GENOME}" ]]; then echo "[ERROR] Genome dir not found: ${BISMARK_GENOME}"; exit 1; fi

OUT_DIR="${OUT_ROOT}/${SAMPLE_ID}"
mkdir -p "${OUT_DIR}"

echo "[INFO] Sample: ${SAMPLE_ID}"
echo "[INFO] Output: ${OUT_DIR}"
echo "[INFO] Genome: ${BISMARK_GENOME}"
echo "[INFO] Trim cores: ${TRIMGALORE_CORES} | Bismark cores: ${BISMARK_CORES}"
echo "[INFO] Inputs: R1=${R1} | R2=${R2}"
echo "[INFO] link_inputs=${LINK_INPUTS} keep_intermediates=${KEEP_INTERMEDIATES}"

# -----------------------------
# Stage inputs
# -----------------------------
R1_BASE="$(basename "${R1}")"
R2_BASE="$(basename "${R2}")"

if [[ "${LINK_INPUTS}" -eq 1 ]]; then
  ln -sf "${R1}" "${OUT_DIR}/${R1_BASE}"
  ln -sf "${R2}" "${OUT_DIR}/${R2_BASE}"
else
  cp -v "${R1}" "${OUT_DIR}/"
  cp -v "${R2}" "${OUT_DIR}/"
fi

# Figure out stems for downstream filenames.
# Supports .fastq.gz, .fq.gz, .fastq, .fq
strip_suffix() {
  local f="$1"
  f="${f%.fastq.gz}"
  f="${f%.fq.gz}"
  f="${f%.fastq}"
  f="${f%.fq}"
  echo "$f"
}

R1_STEM="$(strip_suffix "${R1_BASE}")"
R2_STEM="$(strip_suffix "${R2_BASE}")"

# -----------------------------
# 1) Trim Galore
# -----------------------------
echo "[INFO] Running Trim Galore..."
trim_galore \
  --cores "${TRIMGALORE_CORES}" \
  --gzip \
  --paired \
  "${OUT_DIR}/${R1_BASE}" \
  "${OUT_DIR}/${R2_BASE}" \
  --output_dir "${OUT_DIR}"

TRIM_R1="${OUT_DIR}/${R1_STEM}_val_1.fq.gz"
TRIM_R2="${OUT_DIR}/${R2_STEM}_val_2.fq.gz"

if [[ ! -f "${TRIM_R1}" || ! -f "${TRIM_R2}" ]]; then
  echo "[ERROR] Trim Galore outputs not found:"
  echo "  ${TRIM_R1}"
  echo "  ${TRIM_R2}"
  exit 1
fi

# -----------------------------
# 2) Bismark alignment
# -----------------------------
echo "[INFO] Running Bismark..."
bismark \
  --genome "${BISMARK_GENOME}" \
  -q \
  --parallel "${BISMARK_CORES}" \
  --gzip \
  -1 "${TRIM_R1}" \
  -2 "${TRIM_R2}" \
  --output_dir "${OUT_DIR}"

BAM="${OUT_DIR}/${R1_STEM}_val_1_bismark_bt2_pe.bam"
if [[ ! -f "${BAM}" ]]; then
  echo "[ERROR] Expected BAM not found: ${BAM}"
  exit 1
fi

# -----------------------------
# 3) Deduplicate
# -----------------------------
echo "[INFO] Deduplicating BAM..."
deduplicate_bismark \
  --bam \
  -p \
  --output_dir "${OUT_DIR}" \
  "${BAM}"

DEDUP_BAM="${OUT_DIR}/${R1_STEM}_val_1_bismark_bt2_pe.deduplicated.bam"
if [[ ! -f "${DEDUP_BAM}" ]]; then
  echo "[ERROR] Expected deduplicated BAM not found: ${DEDUP_BAM}"
  exit 1
fi

# -----------------------------
# 4) Methylation extractor
# -----------------------------
echo "[INFO] Running bismark_methylation_extractor..."
bismark_methylation_extractor \
  --paired-end \
  --no_overlap \
  --ignore_r2 2 \
  --gzip \
  --multicore "${BISMARK_CORES}" \
  --bedGraph \
  --zero_based \
  --output "${OUT_DIR}" \
  "${DEDUP_BAM}"

COV_GZ="${OUT_DIR}/${R1_STEM}_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz"
if [[ ! -f "${COV_GZ}" ]]; then
  echo "[ERROR] Expected .cov.gz not found: ${COV_GZ}"
  exit 1
fi

# -----------------------------
# 5) Merge CpGs across strands
# -----------------------------
echo "[INFO] Running coverage2cytosine --merge_CpG..."
coverage2cytosine \
  --merge_CpG \
  --zero_based \
  --genome_folder "${BISMARK_GENOME}" \
  --dir "${OUT_DIR}" \
  -o "${R1_STEM}_val_1_bismark_bt2_pe.deduplicated.bismark.cpgMerged" \
  "${COV_GZ}"

echo "[INFO] Expected final outputs include something like:"
echo "  ${OUT_DIR}/*cpgMerged.CpG_report.merged_CpG_evidence.cov*"

# -----------------------------
# Optional cleanup
# -----------------------------
if [[ "${KEEP_INTERMEDIATES}" -eq 0 ]]; then
  echo "[INFO] Cleaning intermediates..."
  find "${OUT_DIR}" -type f \( \
    -name "*.txt.gz" -o \
    -name "*.bam"   -o \
    -name "*.fq.gz" \
  \) -delete
else
  echo "[INFO] Keeping intermediates (no cleanup)."
fi

echo "[DONE] Finished sample ${SAMPLE_ID}"
