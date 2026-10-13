"""Colour constants for plots.

Used by: human_group_abundance_compare.py, human_corr_landscape.py.
"""

import matplotlib.colors


BLUE = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['royalblue'])
RED = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['crimson'])
GREEN = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['lime'])
WHITE = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['white'])

OURS_GREEN = (0 / 255, 158 / 255, 115 / 255)  # #009E73
NC_PINK = (204 / 255, 121 / 255, 167 / 255)  #CC79A7

ALL_SUM_VIOLET = (114 / 255, 93 / 255, 239 / 255)  # #725DEF
NEG_SUM_SKYBLUE = (102 / 255, 204 / 255, 238 / 255)  # #66CCEE
POS_SUM_MAGENTA = (221 / 255, 33 / 255, 125 / 255)  # #DD217D

EACH_NT_MARIGOLD = (255 / 255, 176 / 255, 13 / 255)  # #FFB00D

BOX_FILLER_BLUE = (0 / 255, 114 / 255, 178 / 255)  # #0072B2
BOX_FILLER_CORAL = (238 / 255, 102 / 255, 119 / 255)  # #EE6677
BOX_FILLER_YELLOW = (240 / 255, 228 / 255, 66 / 255)  # #F0E442
BOX_FILLER_GREEN = (34 / 255, 136 / 255, 34 / 255)  # #228822

LLM_ACCURACY_BLUE = (16 / 255, 85 / 255, 154 / 255)  # #10559A
LLM_PRECISION_TEAL = (64 / 255, 176 / 255, 166 / 255)  # #40B0A6
LLM_RECALL_PURPLE = (81 / 255, 40 / 255, 136 / 255)  # #512888
LLM_F1_CRIMSON = (219 / 255, 16 / 255, 72 / 255)  # #DB1048

# dataviz reference palette: categorical slots 1-2 (validated colorblind-safe pair) + neutrals.
# human_group_abundance_compare.py uses the pair for CTL/SCZ, human_corr_landscape.py for its two
# curves; each curve carries its own legend label, so identity never rests on color alone.
DATAVIZ_BLUE = '#2a78d6'
DATAVIZ_ORANGE = '#eb6834'
DISEASE_COLORS = {'CTL': DATAVIZ_BLUE, 'SCZ': DATAVIZ_ORANGE}
CONTRIBUTION_COLOR = DATAVIZ_BLUE
DIFF_COLOR = DATAVIZ_ORANGE
BASELINE_COLOR = '#c3c2b7'
GRID_COLOR = '#e1e0d9'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
TEXT_MUTED = '#898781'
SURFACE_COLOR = '#fcfcfb'
