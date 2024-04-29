import os
import time
import pickle
import logging
import subprocess


logging.basicConfig(filename='repo_cloner.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


ALLOWED_LICENCES = (
    'MIT License',
    'MIT No Attribution',
    'Apache License 2.0',
    'GNU General Public License v3.0',
    'GNU General Public License v2.0',
    'GNU Affero General Public License v3.0',
    'GNU Lesser General Public License v3.0',
    'GNU Lesser General Public License v2.1',
    'BSD 3-Clause "New" or "Revised" License',
    'Universal Permissive License v1.0',
    'The Unlicense'
)


def process_repo(repo_dir, repo_name):
    counter = 0
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith((".py", ".ipynb", ".txt")):
                os.remove(os.path.join(root, file))
                counter+= 1
    logger.info(f"Deleted {counter} files from {repo_name}")


def clone_repos(pickle_file, directory, delay):
    with open(pickle_file, "rb") as f:
        repos = pickle.load(f)
        del repos['saved_dates']

    for repo_full_name, repo_details in repos.items():
        repo_license = repo_details['license'].get('name', None) if repo_details['license'] else None
        if repo_license not in ALLOWED_LICENCES:
            logger.warning(f"{repo_full_name} licence not on allowed licences list.")
            continue
        repo_dir = f'{directory}/{repo_full_name}'
        repo_name = repo_full_name.split("/")[-1]

        if os.path.exists(repo_dir):
            logger.info(f"Repo {repo_full_name} already cloned.")
        else:
            process = subprocess.run(["git", "clone", f"https://github.com/{repo_full_name}.git", repo_dir], stderr=subprocess.PIPE)

            if process.returncode != 0:
                logger.error(f'Error cloning {repo_full_name}: {process.stderr.decode("utf-8")}')
            else:
                logger.info(f"{repo_full_name} cloned successfully.")
                process_repo(repo_dir, repo_name)

            print(f"{repo_full_name} saved successfully.")
            time.sleep(delay)


if __name__ == "__main__":
    clone_repos("/home/tobiasz/Repos/python-libraries-stats/jupyter-notebook-repos.pickle", "/home/tobiasz/Repos/python-libraries-stats/cloned_repos", 1.5)
