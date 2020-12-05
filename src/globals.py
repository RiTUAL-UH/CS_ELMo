import os

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = None

# OUT-OF-SOURCE DIRECTORIES
CHECKPOINT_DIR  = os.path.join(PROJ_DIR, 'checkpoints')
REPORT_DIR      = os.path.join(PROJ_DIR, 'reports')
DATA_DIR        = os.path.join(PROJ_DIR, 'data')

# SUBFOLDERS OF REPORT
FIGURE_DIR      = os.path.join(REPORT_DIR, 'figures')
HISTORY_DIR     = os.path.join(REPORT_DIR, 'history')
PREDICTIONS_DIR = os.path.join(REPORT_DIR, 'predictions')
ATTENTIONS_DIR  = os.path.join(REPORT_DIR, 'attentions')

# ===========================================================================
REPORTS_DIR = os.path.join(PROJ_DIR, 'reports')
