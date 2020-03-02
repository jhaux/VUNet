import math
import os
import requests
import yaml

from tqdm.auto import tqdm
from zipfile import ZipFile

import VUNet


def _download_file_from_google_drive(id, destination, size=None):
    # taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = _get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    _save_response_content(response, destination, size)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination, size=None):
    CHUNK_SIZE = 32768
    if size is not None:
        total = math.ceil(size / CHUNK_SIZE)
    else:
        total = None

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE),
                          total=total,
                          desc='Downloading Prjoti_J'):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_prjoti(zip_path):
    '''Downloads the tiny Prjoti_J split of the Prjoti Dataset to
    :attr:`store_root`. Also sets up the correct config for this dataset.
    
    Paramters
    ---------
    store_root : str
        Path to the folder, where the dataset is supposed to be stored.
    
    '''

    prjoti_id = '186CE_r0gfgaF6CznBg3zu9_HxioeZvM-'
    _download_file_from_google_drive(prjoti_id, zip_path, size=1589507114)


def extract_prjoti(zip_path):
    '''Extracts the zip and then deletes it.'''

    extract_path = os.path.dirname(zip_path)
    print(f'Extracting content to {extract_path}')

    with ZipFile(zip_path, 'r') as zipObj:
       # Extract all the contents of zip file in different directory
       # zipObj.extractall(extract_path)

       uncompress_size = sum((file.file_size for file in zipObj.infolist()))

       extracted_size = 0

       for file in tqdm(zipObj.infolist(), desc='Unzipping'):
           extracted_size += file.file_size
           percentage = extracted_size * 100/uncompress_size
           zipObj.extract(file, extract_path)

    os.remove(zip_path)


def prep_config(store_root):
    '''Ensures, that the correct data_root parameter is set in the config.'''

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(VUNet.__file__)),
        'configs/prjoti.yaml'
    )
    with open(config_path, 'r') as cf:
        content = yaml.safe_load(cf.read())

    content['data_root'] = store_root

    with open(config_path, 'w+') as cf:
        cf.write(yaml.dump(content))

    meta_path = os.path.join(store_root, 'meta.yaml')
    with open(meta_path, 'r') as mf:
        content = yaml.safe_load(mf.read())

    content['loader_kwargs']['frame']['root'] = store_root
    content['loader_kwargs']['crop']['root'] = store_root

    with open(meta_path, 'w+') as mf:
        content = mf.write(yaml.dump(content))


def ask_store_path(default_root=None):
    '''Interface for entering the path, where the dataset should be stored.'''

    print('======================================')
    print('====== Prjoti_J Dataset Download =====')
    print('======================================')
    print()
    print('It seems you do not have the Prjoti_J dataset on you system. This '
          'dialogue will guide you through the steps necessary to download '
          'it. The dataset needs 1.6GB when extracted and 3.2GB temporarily '
          'during installation.')
    print()
    print('Please enter the absolute path, where you want Prjoti_J to be '
          'stored. Make sure it is a path to a possibly empty folder, which '
          'already exists.')
    print()
    print( '>> ctrl+c to abort')
    print()

    if default_root is not None:
        print(f'>> Press enter without filling in anything to choose '
              f'`{default_root}`.')

    store_url = str(input())
    if store_url == '':
        store_url = default_root

    store_url = os.path.join(store_url, 'Prjoti_J.zip')

    print()
    print(f'File will be saved to `{store_url}`.')

    return store_url


def prjoti_installer(default_root=None):
    '''Donwloads and installs the Prjoti_J dataset.'''

    zip_path = ask_store_path(default_root)

    download_prjoti(zip_path)
    extract_prjoti(zip_path)

    root_path = os.path.dirname(zip_path)
    prep_config(root_path)

    return root_path


if __name__ == '__main__':
    prjoti_installer()
    # store_url = '/home/jhaux/Downloads/Prjoti_J/Prjoti_J.zip'
    # extract_prjoti(store_url)
