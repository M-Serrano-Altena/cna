"""
Command-line utility for transferring files between a local machine and a remote cluster.

This script provides a flexible interface to upload or download files to/from
preconfigured computing clusters. It wraps the `transfer_files_cluster` function
and exposes its functionality via command-line arguments.

Features:
- Supports multiple clusters (e.g. csm, snellius variants, hipster)
- Upload or download modes
- Transfer specific files or all files in a directory
- Configurable local and remote directories
- Optional overwrite of existing files
- Uses a credentials file for authentication

Usage:
    # Download specific files from the cluster to the current local directory
    python cluster_transfer.py --transfer_mode download --files file1.txt file2.txt --cluster snellius_cna
    → Downloads "file1.txt" and "file2.txt" from the remote directory on the cluster into the current local directory.

    # Download all files from a remote directory into a specified local directory
    python cluster_transfer.py --transfer_mode download --local_dir /local/directory --remote_dir /remote/directory --cluster snellius_cna
    → Copies all files from "/remote/directory" on the cluster to "/local/directory" locally. Existing files in the local directory will be preserved unless the --overwrite flag is set.

    # Upload all files from a local directory to a remote directory on the cluster
    python cluster_transfer.py --transfer_mode upload --local_dir /local/directory --remote_dir /remote/directory --cluster snellius_cna
    → Sends all files from "/local/directory" to "/remote/directory" on the cluster. Existing files in the remote directory will be preserved unless the --overwrite flag is set.

Arguments:
    --cluster : str
        Target cluster. Must be one of ['csm', 'snellius_polymer', 'snellius_cna', 'hipster'].
        Default is 'snellius_cna'.

    --transfer_mode : str
        Mode of transfer. Must be either 'upload' or 'download'. This argument is required explicitly.

    --local_dir : str
        Path to the local directory. Defaults to the current directory.

    --remote_dir : str
        Path to the remote directory on the cluster. Defaults to CLUSTER_BASE_DIR.

    --cluster_credentials_file : str
        Path to the JSON file containing cluster login credentials.
        Defaults to ~/.config/clusters/cluster_credentials.json.

    --files : list of str or "all"
        Specific files to transfer. Defaults to ["all"], meaning all files will be transferred.

    --overwrite, -o : bool
        If set, existing files at the destination will be overwritten. Otherwise, existing files will be preserved and the transfer will be skipped for those files.

Raises:
    ValueError:
        If `transfer_mode` is not explicitly set to 'upload' or 'download'.

Notes:
    - The credentials file must contain valid authentication details for the selected cluster.
    - If `files="all"`, the entire contents of the specified directory are transferred.

*Author: Marc Serrano Altena,*
*Date: 2026-05-06*
"""

import argparse
from output_file_utils import transfer_files_cluster, CLUSTER_BASE_DIR
import os

CLUSTER_CREDENTIALS_DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".config", "clusters", "cluster_credentials.json")

def main() -> None:
    clusters = ['csm', 'snellius_polymer', 'snellius_cna', 'hipster']
    default_cluster = 'snellius_cna'

    parser = argparse.ArgumentParser(description="Transfer files to/from a cluster")

    # Cluster configuration parameters
    parser.add_argument('--cluster', choices=clusters, default=default_cluster,
                        help='Cluster to use for running the simulation. Options are "csm", "snellius_polymer", "snellius_cna", or "hipster".')
    parser.add_argument('--transfer_mode', choices=['upload', 'download',], default='',
                        help='Mode of file transfer: "upload" to send files to the cluster, "download" to retrieve files from the cluster.')
    parser.add_argument('--local_dir', type=str, default=".",
                        help='Local directory to use for file transfers. Default is current directory.')
    parser.add_argument('--remote_dir', type=str, default=CLUSTER_BASE_DIR,
                        help='Remote directory on the cluster to use for file transfers. Default is the cluster base directory.')
    parser.add_argument('--cluster_credentials_file', type=str, default=CLUSTER_CREDENTIALS_DEFAULT_PATH,
                        help='Path to cluster credentials file. Default is ~/.config/clusters/cluster_credentials.json.')
    parser.add_argument('--files', nargs='*', default=["all"],
                        help='List of files to transfer. Can be a single file or multiple files. Default is "all", which means all files will be transferred.')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing output file if it exists.')
    parser_args = parser.parse_args()

    cluster = parser_args.cluster
    transfer_mode = parser_args.transfer_mode
    local_dir = parser_args.local_dir
    remote_dir = parser_args.remote_dir
    cluster_credentials_file = parser_args.cluster_credentials_file
    files = parser_args.files
    overwrite = parser_args.overwrite

    transfer_mode_options = ["upload", "download"]
    if transfer_mode not in transfer_mode_options:
        raise ValueError(f"The transfer_mode must be explicitely specified as one of {transfer_mode_options}")

    if files == ["all"]:
        files = "all"

    # Call the transfer function with the parsed arguments
    transfer_files_cluster(
        cluster_name=cluster,
        transfer_mode=transfer_mode,
        files=files,
        local_directory=local_dir,
        remote_directory=remote_dir,
        cluster_credentials_file=cluster_credentials_file,
        overwrite=overwrite,
    )

if __name__ == "__main__":
    main()