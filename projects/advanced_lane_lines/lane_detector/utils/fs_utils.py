import os
import urllib.request
import zipfile


def download_file(url, download_to_dir, delete_old=False):
    '''
    Given a URL and a path to directory, downloads the file the URL points to
    if the file already exists and delete_old is set to True, deletes the file
    before downloading

    Args:
        url (str) : Public URL pointing to a file
        download_to_dir (str) : directory to download the file to. If it doesn't exist
                                it is created.
        delete_old (bool) : Default False, If set to true & the file alredy exists in path
                            its deleted before downloading a new copy
    Returns:
        filepath (str) : Path where the file was downloaded to.
    '''
    if not os.path.exists(download_to_dir):
        os.makedirs(download_to_dir)
    download_path = os.path.join(download_to_dir, url[url.rfind('/') + 1:])

    if os.path.exists(download_path):
        if delete_old:
            os.remove(download_path)
        else:
            return download_path

    urllib.request.urlretrieve(url, download_path)
    return download_path


def extract_zip(filepath, extract_to_dir):
    '''
    Given a zip file path & a target directory extracts the zip contents 
    into the target directory
    Args:
     filepath (str) : Local path to the zipfile
     extract_to_dir (str) : Target path to the zipfile contents.
                            If doesn't exist, its created
    '''
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    with zipfile.ZipFile(filepath, 'r') as ref:
        ref.extractall(extract_to_dir)
