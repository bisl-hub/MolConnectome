"""Whether a flagged connection's direction agrees with its sender region's abundance change.

Used by: human_prioritize_drivers.py.
"""


def _is_direction_consistent(status: str, abundance_direction: str, abundance_significance: str) -> bool:
    '''A connection is only drawn when its connectivity status agrees with the
    sender region's abundance change: either the abundance moved the same way,
    or it moved the other way but not significantly so.'''
    if status == 'hyperactive':
        return (abundance_direction == 'abundance elevated in SCZ'
                or (abundance_direction == 'abundance decreased in SCZ'
                    and abundance_significance == 'not_significantly_different'))
    if status == 'hypoactive':
        return (abundance_direction == 'abundance decreased in SCZ'
                or (abundance_direction == 'abundance elevated in SCZ'
                    and abundance_significance == 'not_significantly_different'))
    return False
