import os
from pathlib import Path
import numpy as np
from detector import detect_and_crop

def generate_cropped_videos(path_source: str, path_output:str, amount: int = None, verbose:bool = True, **kwargs) -> None:
    files = os.listdir(path_source)
    N = len(files)
    amount = N if not amount else amount

    if amount != N:
        files = np.array(files)
        files = np.random.choice(files, size=amount, replace=False).tolist()

    if not os.path.exists(path_output):
        if verbose: print(f'Generating folder: {path_out}')
        os.mkdir(path_output)


    for i, file in enumerate(files):
        if verbose: print(f"\r[{i + 1} / {amount}] Processing file: {file} ", end='')
        path_to_file = os.path.join(path_source, file)
        path_to_save = os.path.join(path_out, Path(file).stem)
        detect_and_crop(path_to_file, path_to_save, **kwargs)

    if verbose: print()


if __name__ == "__main__":
    path_source = "./data"
    path_out = "./out"
    generate_cropped_videos(path_source, path_out, nth_frame=10)


