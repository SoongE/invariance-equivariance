from __future__ import print_function

import torch





def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
