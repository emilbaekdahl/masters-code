import pathlib
import tarfile
import typing as tp
import zipfile

import tqdm


def decompress_tar(path, dest=None, keep=False):
    with tarfile.open(path, "r:gz") as tar_file:
        for file in tqdm.tqdm(tar_file.getmembers(), desc="Decompressing"):
            tar_file.extract(file, path=dest)

    if keep is False:
        path.unlink()

    return dest


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
