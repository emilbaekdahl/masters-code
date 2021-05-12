import os.path
import pathlib
import typing as tp

import requests as rq
import tqdm


def download_file(url: str, dest_folder: tp.Union[str, pathlib.Path]) -> pathlib.Path:
    """Downloads a single file to a destination folder.

    Paramters:
        url: Location of the file to download.
        dest_folder: Folder to put the downloaded file in.

    Returns:
        Path to the downloaded file.
    """
    if not isinstance(dest_folder, pathlib.Path):
        dest_folder = pathlib.Path(dest_folder)

    dest_folder.mkdir(parents=True, exist_ok=True)

    file_name = os.path.basename(url)
    dest = dest_folder / file_name

    if dest.exists():
        return dest

    response = rq.get(url, stream=True)

    file_size = int(response.headers["Content-Length"])

    with open(dest, "wb") as file, tqdm.tqdm(
        desc="Downloading",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=file_size,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            write_size = file.write(chunk)
            pbar.update(write_size)

    return dest
