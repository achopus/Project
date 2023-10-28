import csv

def load_segments_from_file(path: str) -> list[list[int, int]]:
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        line = None
        for l in reader: line = l
    segments = []
    for s in line:
        seg = [int(x) for x in list(s[1:-1].split(sep=','))]
        segments.append(seg)
    return segments
