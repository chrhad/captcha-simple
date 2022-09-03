from typing import List

LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_IDXS = dict([(c,i) for (i,c) in enumerate(LABELS)])

def to_label_str(label_ids: List[int]) -> List[str]:
    return [LABELS[i] for i in label_ids]
