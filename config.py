import os

# --- Paths ---
# Use the directory where this script is located as the base
import pathlib
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = str(pathlib.Path(CONFIG_DIR).parent)
SHARED_HISTONE_DIR = str(pathlib.Path(RELEASE_DIR).parent)

# Use the shared raw dataset. This release keeps its own processed cache because
# the older shared processed files were built with a different mark set.
# Updated to use the 400 cell-line dataset. The old data_gold_154 path was
# inaccessible and only contained ~150 cell lines. Using the full set improves
# generalization of the shared CNN-Mamba encoder.
DATA_DIR = os.environ.get("HISTONE_DATA_DIR", "./data")
PROCESSED_DIR = os.path.join(CONFIG_DIR, "processed")
RUNS_DIR = os.path.join(CONFIG_DIR, "runs")
PLOTS_DIR = os.path.join(RUNS_DIR, "plots")

# Note: FASTA and GTF usually need absolute external paths
FASTA_PATH = os.environ.get("HISTONE_FASTA_PATH", "./hg38/hg38.fa")
GTF_PATH = os.environ.get("HISTONE_GTF_PATH", "./hg38/hg38.knownGene.gtf")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Data Settings ---
# Marks selected for TF binding prediction:
#   ENHANCING marks  → chromatin open, TFs freely bind
#   INHIBITING marks → chromatin compacted, TFs blocked
TARGET_MARKS = [
    "H3K27ac",   # ENHANCES TF binding - active enhancers + promoters
    "H3K4me3",   # ENHANCES TF binding - active promoters (sharp, sequence-driven)
    "H3K27me3",  # INHIBITS TF binding - Polycomb repression
    "H3K9me3",   # INHIBITS TF binding - constitutive heterochromatin
]
NUM_MARKS = len(TARGET_MARKS)  # = 4

# 8192bp windows capture broad enhancer/repressor context for H3K27ac and H3K27me3
WINDOW_SIZE = 32768
HALF_WINDOW = WINDOW_SIZE // 2

# Top most variable genes to use as cell-line RNA fingerprint
RNA_NUM_TOP_GENES = 4000
# Since we are peak-centered, there is no single "local gene". 
# The model just takes the global fingerprint.
RNA_INPUT_DIM = RNA_NUM_TOP_GENES

# --- Model Architecture Settings ---
# DNA Encoder: Multi-Resolution CNN Stem (3 branches) + Mamba-2 SSM
# Total stride = 64 → 32768/64 = 512 tokens for Mamba

D_MODEL = 256         # Proven at hist3 (0.71+ Pearson). 512 was 4x heavier for no gain.
N_LAYERS = 4          # 4 BiMamba layers on 512 tokens is plenty of context modeling
D_STATE = 32          # SSM hidden state dimension (Mamba-2)
D_CONV = 4            # Local convolution kernel in Mamba block

MAMBA_EXPAND = 2      # Inner dimension expansion factor
# Increased to 0.3: stronger regularisation to prevent mark-conflict overfitting
DROPOUT = 0.3

# RNA Encoder (MLP)
RNA_HIDDEN_DIM = 512  # Keep at 512 - RNA compression is independent of DNA width

# Cross-Attention: DNA positions query RNA tokens for position-specific
# cell-type conditioning. K tokens are projected from the RNA embedding.
NUM_RNA_TOKENS = 8       # Number of RNA "pseudo-tokens" for cross-attention
CROSS_ATTN_HEADS = 4     # 256 / 4 = 64 dims per head

# Set True for an RNA ablation experiment. This zeros the RNA vector during
# train/validation/evaluation so performance can be compared against the
# full RNA-conditioned model.
RNA_ABLATION = False

# Mark-specific context pooling: the shared encoder sees the full WINDOW_SIZE,
# but each mark head pools over a different central span of encoded tokens.
MARK_CONTEXT_BP = {
    "H3K27ac": 16384,   # Widened from 8192 - enhancers are diffuse, need more context
    "H3K4me3": 4096,
    "H3K27me3": 32768,
    "H3K9me3": 16384,
}

# Multitask Head
HEAD_HIDDEN_DIM = 256   # Proportional to D_MODEL

# --- Training Settings ---
BATCH_SIZE = 256      # Increased to 256 since GPU has ~40GB free (uses ~24GB)
GRAD_ACCUM_STEPS = 4  # Effective batch size = 256 × 4 = 1024
# LR 3e-4: original setting that achieved best Pearson 0.7153
LEARNING_RATE = 3e-4
# Stronger L2 to fight overfitting (train loss diverged from val at ep5+)
WEIGHT_DECAY = 5e-2
EPOCHS = 50
# Original warmup
WARMUP_STEPS = 3000
# Longer patience: previous run peaked at ep4 with patience=7, giving only 11 epochs total
EARLY_STOPPING_PATIENCE = 10
MAX_TRAIN_BATCHES = None

# V7 Data Loading Settings
PEAK_FRAC = 0.50  # 50% peaks, 50% background to balance signal and boundary learning
BG_MARGIN = 20000  # Margin to exclude peaks from background windows (covers broad H3K27me3/H3K9me3)
AUGMENT_RC = True
# Wider jitter: 500bp was too small (1.5% of window); 2000bp forces true translation invariance
AUGMENT_OFFSET_MAX_BP = 2000
MAX_PEAKS_PER_MARK_PER_CELL = 3000 # 73% coverage of sparsest mark; sparse marks auto-oversampled via replace=True

# Loss weighting: alpha * Huber + (1-alpha) * Pearson
LOSS_ALPHA = 0.4

# Track Head settings
# TRACK_BIN_SIZE must equal CNN total stride (product of CNN_STRIDES).
# With CNN_STRIDES=[1,4,4,4] → stride=64, so each output token = 64 bp.
TRACK_BIN_SIZE = 64           # bp per output bin; must match CNN stride product
NUM_TRACK_BINS = WINDOW_SIZE // TRACK_BIN_SIZE  # = 512 bins
# Weight for auxiliary track BCE loss relative to main scalar loss.
# Reduced from 1.0: track was consuming 59% of training gradient, causing
# the shared encoder to memorise spatial patterns at the expense of scalar Pearson.
TRACK_LOSS_WEIGHT = 0.3

# Presence/absence threshold in log2(signal + 1) space for derived metrics
PRESENCE_THRESHOLD = 0.5

# Genome Cache
GENOME_CACHE_PATH = os.path.join(PROCESSED_DIR, "genome_hg38.bin")

# Dataloader
NUM_WORKERS = 16  # 256 cores available; file_system sharing strategy prevents segfault
PIN_MEMORY = True
