import matplotlib.colors
from matplotlib import colormaps

TITLE_SIZE = 23
SUBTITLE_SIZE = 16
CAPTION_SIZE = 9


human_network_style_dict = {
    'PFC':  {'edgecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#FAB0E4')),
              'labelcolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#000000')),
              'facecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#FAB0E4')),
              },
    'HP':   {'edgecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#D0BBFF')),
              'labelcolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#000000')),
              'facecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#D0BBFF')),
              },
    'CC':   {'edgecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#FFFEA3')),
              'labelcolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#000000')),
              'facecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#FFFEA3')),
              },
    'CN':   {'edgecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#8DE5A1')),
              'labelcolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#000000')),
              'facecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#8DE5A1')),
              },
    'NAc':  {'edgecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#FFB482')),
              'labelcolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#000000')),
              'facecolor' : matplotlib.colors.to_hex(matplotlib.colors.to_rgb('#FFB482')),
              },
}
