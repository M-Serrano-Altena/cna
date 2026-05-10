import os
from pathlib import Path
import stat
from typing import Any, Callable, Union, Generator, TYPE_CHECKING
import json
import getpass
from contextlib import contextmanager
from tqdm import tqdm

if TYPE_CHECKING:
    from paramiko import SFTPClient

CLUSTER_BASE_DIR = "cna"

def transfer_files_cluster(
    cluster_name: str,
    transfer_mode: str,
    files: Union[list[str], str, None] = None,
    remote_directory: str = "",
    local_directory: str = "",
    cluster_credentials_file: str = os.path.join(os.path.expanduser("~"), ".config", "clusters", "cluster_credentials.json"),
    overwrite: bool = False
) -> None:
    """
    Transfer files between the local machine and a cluster using SFTP.

    Args:
        cluster_name (str): Name of the cluster (used to load credentials).
        transfer_mode (str): Either "download" or "upload".
        files (list[str] | str | None): List of files or directories to transfer, or "all".
        args_dict (dict | None): Simulation arguments to generate file paths.
        varied_args (dict | None): Arguments to vary for batch transfers.
        remote_directory (str): Remote directory on the cluster.
        local_directory (str): Local directory for file transfer.
        overwrite (bool): If True, overwrite existing files.

    Raises:
        ValueError: If required arguments are missing or invalid.
    """

    if files is None:
        raise ValueError("Either args_dict must be provided or both remote_directory and local_directory must be specified.")

    if CLUSTER_BASE_DIR not in remote_directory:
        remote_directory = os.path.join(CLUSTER_BASE_DIR, remote_directory)

    cluster_credentials = load_cluster_credentials(cluster_name, cluster_credentials_file)

    hostname = cluster_credentials.get("hostname")
    hostname = repeat_while_invalid(
        hostname,
        input_string=f"Enter hostname for {cluster_name}: ",
        error_string="Hostname cannot be empty. Please enter a valid hostname."
    )
    port = cluster_credentials.get("port", 22)
    port = repeat_while_invalid(
        port,
        input_string=f"Enter port for {cluster_name} (default 22): ",
        error_string="Port must be a valid integer.",
        cast=int
    )
    username = cluster_credentials.get("username")
    username = repeat_while_invalid(
        username,
        input_string=f"Enter username for {cluster_name}: ",
        error_string="Username cannot be empty. Please enter a valid username."
    )
    password = cluster_credentials.get("password")
    password = repeat_while_invalid(
        password,
        input_func=getpass.getpass,
        input_string=f"Enter password for {cluster_name}: ",
        error_string="Password cannot be empty. Please enter a valid password."
    )

    if not files:
        raise ValueError("No files provided for transfer.")

    if transfer_mode not in ["download", "upload"]:
        raise ValueError("Invalid transfer mode. Use 'download' or 'upload'.")

    if transfer_mode == "download":
        download_files_from_server(
            files=files,
            remote_directory=remote_directory,
            local_directory=local_directory,
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            overwrite=overwrite
        )
    
    elif transfer_mode == "upload":
        upload_files_to_server(
            files=files,
            remote_directory=remote_directory,
            local_directory=local_directory,
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            overwrite=overwrite
        )


def download_files_from_server(
    files: Union[list[str], str],
    remote_directory: str,
    local_directory: str,
    hostname: str,
    port: int,
    username: str,
    password: str,
    overwrite: bool = False
) -> None:
    """
    Downloads a list of files from a remote server to a local directory using SFTP.

    Args:
        files (list[str]): List of filenames to download.
        remote_directory (str): Path to the directory on the remote server.
        local_directory (str): Path to the local directory where files will be saved.
        hostname (str): Hostname or IP address of the remote server.
        port (int): Port number for the SFTP connection.
        username (str): Username for authentication.
        password (str): Password for authentication.
        overwrite (bool): If True, overwrite existing files in the local directory.
    """
    with sftp_connection(hostname, port, username, password) as sftp:
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        try:
            remote_files = set(list_remote_files_recursive(sftp, remote_directory))
        except IOError:
            print(f"Remote directory does not exist: {remote_directory}")
            return
        
        # If 'all' is specified, we assume all files in the remote directory should be downloaded
        if files == "all":
            files = list(remote_files)
            files = remove_base_dir_from_paths(files, remote_directory)
        elif isinstance(files, str):
            files = [files]

        # If files are specified, we need to ensure they exist in the remote directory
        # Also expand directories to include all files within them
        files = remove_base_dir_from_paths(files, local_directory)
        expanded_files = []
        for path in files:
            remote_path = os.path.join(remote_directory, path)
            for remote_filename in list_remote_files_recursive(sftp, remote_path):
                remote_filepath = os.path.join(remote_path, remote_filename)
                expanded_files.append(remote_filepath)

        # Remove base remote directory from paths
        files = remove_base_dir_from_paths(expanded_files, remote_directory)

        files = sorted(files, key=lambda x: x.lower())

        # Download the files
        for file in files:
            if file not in remote_files:
                print(f"File {file} does not exist in remote directory {remote_directory}")
                continue

            remote_filepath = os.path.join(remote_directory, file)
            local_filepath = os.path.join(local_directory, file)
            if os.path.isfile(local_filepath) and not overwrite:
                print("========================================================================")
                print(f"Local filepath {local_filepath} already exists. Skipping download. Use --overwrite or -o to overwrite.")
                print("========================================================================")
                continue

            try:
                print(f"Retrieving {remote_filepath} to {local_filepath}")
                local_path = os.path.dirname(local_filepath)
                os.makedirs(local_path, exist_ok=True)
                file_size = sftp.stat(remote_filepath).st_size
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(file), ncols=80) as pbar:
                    def progress_callback(bytes_transferred: int, total_bytes_left_to_transfer: int) -> None:
                        pbar.update(bytes_transferred - pbar.n)

                    sftp.get(remote_filepath, local_filepath, callback=progress_callback)
            except IOError as e:
                print(f"Failed to retrieve {remote_filepath}: {e}")

        print("All files retrieved successfully.")


def upload_files_to_server(
    files: Union[list[str], str],
    remote_directory: str,
    local_directory: str,
    hostname: str,
    port: int,
    username: str,
    password: str,
    overwrite: bool = False
) -> None:
    """
    Uploads a list of files to a remote server using SFTP.

    Args:
        files (list[str]): List of local file paths to upload.
        remote_directory (str): Path to the directory on the remote server.
        local_directory (str): Path to the local directory containing files to upload.
        hostname (str): Hostname or IP address of the remote server.
        port (int): Port number for the SFTP connection.
        username (str): Username for authentication.
        password (str): Password for authentication.
        overwrite (bool): If True, overwrite existing files in the remote directory.
    """
    with sftp_connection(hostname, port, username, password) as sftp:
        # Ensure remote directory exists
        mkdir_p(sftp, remote_directory)
        
        local_files = set(local_expand_to_all_files([local_directory]))

        # If 'all' is specified, we assume all files in the local directory should be uploaded
        if files == "all":
            files = list(local_files)
        elif isinstance(files, str):
            files = [files]

        files = local_expand_to_all_files(files)
        files = remove_base_dir_from_paths(files, local_directory)
        files = sorted(files, key=lambda x: x.lower())
        local_files = set(remove_base_dir_from_paths(list(local_files), local_directory))

        for file in files:
            if file not in local_files:
                print(f"File {file} does not exist in local directory {local_directory}")
                continue

            local_filepath = os.path.join(local_directory, file)
            remote_filepath = os.path.join(remote_directory, file)

            if remote_exists(sftp, remote_filepath) and not overwrite:
                print("========================================================================")
                print(f"Remote file {remote_filepath} already exists. Skipping upload. Use --overwrite or -o to overwrite.")
                print("========================================================================")
                continue

            mkdir_p(sftp, os.path.dirname(remote_filepath))

            print(f"Transferring {local_filepath} to {remote_filepath}")
            file_size = os.path.getsize(local_filepath)
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(local_filepath), ncols=80) as pbar:
                def progress_callback(bytes_transferred: int, total_bytes_left_to_transfer: int) -> None:
                    pbar.update(bytes_transferred - pbar.n)

                sftp.put(local_filepath, remote_filepath, callback=progress_callback)

        print("All files transferred successfully.")


@contextmanager
def sftp_connection(
    hostname: str,
    port: int,
    username: str,
    password: str
) -> Generator["SFTPClient", None, None]:
    """
    Context manager for establishing an SFTP connection using Paramiko.

    Args:
        hostname (str): The hostname or IP address of the SFTP server.
        port (int): The port number to connect to on the SFTP server.
        username (str): The username to authenticate with.
        password (str): The password to authenticate with.

    Yields:
        paramiko.SFTPClient: An active SFTP client object for file operations.

    Example:
        with sftp_connection('example.com', 22, 'user', 'pass') as sftp:
            sftp.get('remote_path', 'local_path')

    Ensures that the SFTP and SSH connections are properly closed after use.
    """
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)
    sftp = ssh.open_sftp()

    try:
        yield sftp  # This "hands over" control to the with-block
    finally:
        sftp.close()
        ssh.close()

def remote_exists(sftp, path: str) -> bool:
    """
    Check if a remote file or directory exists on the SFTP server.

    Args:
        sftp: An active SFTP client object (e.g., from paramiko).
        path (str): The remote file or directory path to check.

    Returns:
        bool: True if the remote file or directory exists, False otherwise.
    """
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False
    
def mkdir_p(sftp, remote_directory: str) -> None:
    """
    Recursively create directories on the remote server like `mkdir -p`.
    """
    dirs = []
    while remote_directory not in ('/', ''):
        dirs.insert(0, remote_directory)
        remote_directory = os.path.dirname(remote_directory)

    for dir_ in dirs:
        try:
            sftp.stat(dir_)
        except FileNotFoundError:
            sftp.mkdir(dir_)
    
def load_cluster_credentials(cluster_name: str, credentials_file: str=os.path.join(os.path.expanduser("~"), ".config", "clusters", "cluster_credentials.json")) -> dict:
    """
    Load cluster credentials from a JSON file.

    Args:
        cluster_name (str): The name of the cluster.

    Returns:
        dict: A dictionary containing the cluster credentials.
    """
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file {credentials_file} not found.")

    credentials = load_object_from_json_file(credentials_file)
    if not isinstance(credentials, dict):
        raise ValueError(f"Credentials file {credentials_file} is not a valid JSON object.")

    if cluster_name not in credentials:
        raise ValueError(f"Cluster '{cluster_name}' not found in credentials file.")

    return credentials[cluster_name]

def list_remote_files_recursive(sftp, remote_path: str, remove_remote_path_from_files: bool = True) -> list[str]:
    """
    Recursively lists all files in a remote directory using SFTP.
    
    This function traverses a remote directory structure and collects all file paths
    found within it, including files in nested subdirectories. The function can
    optionally remove the base remote path from the returned file paths.
    
    Args:
        sftp: An active SFTP client object (e.g., from paramiko).
        remote_path (str): The remote directory path to recursively traverse.
        remove_remote_path_from_files (bool, optional): If True, removes the base
            remote_path from the returned file paths. If False, returns absolute
            paths. Defaults to True.
    
    Returns:
        list[str]: A list of file paths found in the remote directory structure.
            If remove_remote_path_from_files is True, paths are relative to the
            remote_path. If False, paths are absolute.
    
    Raises:
        IOError: If the remote_path cannot be accessed or if there are permission
            issues with any subdirectories.
    
    Example:
        >>> with sftp_connection('server.com', 22, 'user', 'pass') as sftp:
        ...     files = list_remote_files_recursive(sftp, '/remote/dir')
        ...     print(files)
        ['file1.txt', 'subdir/file2.txt', 'subdir/nested/file3.txt']
    """
    
    files = []

    def _recursive_list(path):
        try:
            # Get the stat for this path
            path_stat = sftp.stat(path)

            # If it's a file, add and return
            if not stat.S_ISDIR(path_stat.st_mode):
                files.append(path)
                return

            # Otherwise, it's a directory: recurse
            for item in sftp.listdir_attr(path):
                item_path = os.path.join(path, item.filename)
                if stat.S_ISDIR(item.st_mode):
                    _recursive_list(item_path)
                else:
                    files.append(item_path)

        except IOError as e:
            print(f"Could not access {path}: {e}")

    print("getting all files recursively...")
    _recursive_list(remote_path)
    if remove_remote_path_from_files:
        files = remove_base_dir_from_paths(files, remote_path)

    print("all files found!")
    return files

def remove_base_dir_from_paths(files: list[str], base_dir: str) -> list[str]:
    """
    Remove the base directory from a list of file paths.
    
    This function takes a list of file paths and removes the base directory prefix
    from each path if it exists as a contiguous subset. If the base directory is
    not found as a prefix, the original path is returned unchanged.
    
    Args:
        files (list[str]): List of file paths to process.
        base_dir (str): The base directory path to remove from the file paths.
            
    Returns:
        list[str]: List of file paths with the base directory removed if it was
            a prefix, otherwise the original paths unchanged.
            
    Example:
        >>> files = ['/path/to/file1.txt', '/path/to/subdir/file2.txt']
        >>> base_dir = '/path/to'
        >>> remove_base_dir_from_paths(files, base_dir)
        ['file1.txt', 'subdir/file2.txt']
    """
    
    new_files = []
    for i, file in enumerate(files):
        if is_contiguous_subset(list(Path(base_dir).parts), list(Path(file).parts)):
            new_files.append(os.path.relpath(file, base_dir))
        else:
            new_files.append(file)

    return new_files


def local_expand_to_all_files(paths: Union[list[str], str]) -> list[str]:
    """
    Recursively expands a list of file and directory paths to a flat list of all file paths.

    Given a list of file and/or directory paths, this function traverses any directories,
    collecting all file paths contained within them (including nested subdirectories).
    The result is a list of all file paths found.

    Args:
        paths (Union[list[str], str]): A list of file and/or directory paths, or a single path as a string.

    Returns:
        list[str]: A flat list containing the paths of all files found.
    """
    expanded_files = []

    if isinstance(paths, str):
        paths = [paths]

    while paths:
        path = paths.pop()
        if os.path.isdir(path):
            for entry in os.listdir(path):
                full_entry = os.path.join(path, entry)
                paths.append(full_entry)
        elif os.path.isfile(path):
            expanded_files.append(path)

    expanded_files = sorted(expanded_files, key=lambda x: x.lower())
    return expanded_files


def load_object_from_json_file(filename: str) -> Any:
    """Loads an object from a JSON file

    Args:
        filename (str): The filename to load from.

    Returns:
        Any: The loaded object.
    """
    filename = add_missing_extension(filename=filename, extension=".json")
    with open(filename, "r") as file:
        return json.load(file)
    
def add_missing_extension(filename: str, extension: str) -> str:
    """
    Add an extension to a filename if the extension is missing
    or replace the existing extension with the new one.

    Args:
        filename (str): The filename.
        extension (str): The extension to add if missing.

    Returns:
        str: The filename with the extension.
    """
    base, existing_extension = os.path.splitext(filename)
    if not existing_extension:
        filename = filename + extension
    elif existing_extension != extension:
        filename = base + extension

    return filename

def repeat_while_invalid(
    input_val: Any,
    condition: Callable[[Any], bool] = lambda x: bool(x),  # Default condition checks if input is truthy
    input_func: Callable[[str], str] = input,
    input_string: str = "Enter value: ",
    error_string: str = "Invalid. Please enter a valid value.",
    cast: Union[type, None] = None,
) -> Any:
    """
    Repeatedly prompt the user for input until a valid value is provided.

    Args:
        input_val (Any): Initial value to check.
        condition (Callable[[Any], bool]): Function that returns True if the input is valid.
        input_func (Callable[[str], str], optional): Function to get user input (default: built-in input).
        input_string (str, optional): Prompt string for user input.
        condition_string (str, optional): Message to display when input is invalid.

    Returns:
        Any: Validated input value.
    """
    while not condition(input_val):
        print(error_string)
        input_val = input_func(input_string)
        if cast is not None:
            try:
                input_val = cast(input_val)
            except ValueError:
                print(f"Could not cast input to {cast.__name__}. Please try again.")
                continue
            
    return input_val

def is_contiguous_subset(sub: list, main: list) -> bool:
    """
    Check if 'sub' is a contiguous subset of 'main'.

    A contiguous subset means that all elements of 'sub' appear in 'main' in the same order and without any gaps,
    i.e., there exists an index i such that main[i:i+len(sub)] == sub. The elements must be consecutive in 'main'.

    Args:
        sub (list): The candidate contiguous subset.
        main (list): The main list to check within.

    Returns:
        bool: True if 'sub' is a contiguous subset of 'main', False otherwise.
    """
    n, m = len(main), len(sub)
    if m == 0:
        return True
    if m > n:
        return False

    for i in range(n - m + 1):
        if main[i:i + m] == sub:
            return True
    return False