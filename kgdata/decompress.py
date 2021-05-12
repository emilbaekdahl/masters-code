import pathlib
import typing as tp
import zipfile

import tqdm


def decompress_zip(
    path: tp.Union[str, pathlib.Path], dest=None, keep=False
) -> pathlib.Path:
    """
    Parameters:
        path: Location of the zip file to decompress.

    Returns:
        Path to the decompressed folder.
    """
    with zipfile.ZipFile(path, "r") as zip_file:
        for file in tqdm.tqdm(zip_file.namelist(), desc="Decompressing"):
            zip_file.extract(file, dest)

    if keep is False:
        path.unlink()

    return path.with_suffix("")
