import os
import time
import pickle
import httpx
import logging
from typing import Dict
from datetime import date, timedelta, datetime


LANGUAGE = "python"


logging.basicConfig(filename='repo_collector.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_repos(filename: str) -> Dict:
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {"saved_dates": []}


def save_repos(repos: Dict, filename: str) -> None:
    with open(filename, "wb") as f:
        pickle.dump(repos, f)


def fetch_repositories(year: int, month: int, day: int, token: str) -> Dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}"
    }
    url = (
        f"https://api.github.com/search/repositories?"
        f"q=created:{year}-{month:02d}-{day:02d}+"
        f"language:{LANGUAGE}+stars:>100&"
        f"sort=stars&order=desc&per_page=100"
    )

    with httpx.Client() as client:
        response = client.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch data for {year}-{month:02d}-{day:02d}: {response.text}")
            return {}
        
        results = response.json().get("items", [])
        return {
            repo["full_name"]: {
                "id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "html_url": repo["html_url"],
                "created_at": repo["created_at"],
                "updated_at": repo["updated_at"],
                "size": repo["size"],
                "stargazers_count": repo["stargazers_count"],
                "topics": repo["topics"],
                "watchers": repo["watchers"],
                "license": repo["license"]
            }
            for repo in results
        }


def main():
    start_date = date(2020, 1, 1)
    end_date = datetime.now().date()
    token = os.getenv("GITHUB_TOKEN")
    filename = f"{LANGUAGE}-repos.pickle"
    repos = load_repos(filename)
    start_date = max(repos["saved_dates"]) if repos["saved_dates"] else start_date
    logger.info(f"Loaded repos pickle. Start date: {start_date}")


    current_date = start_date 
    while current_date < end_date:
        current_date += timedelta(days=1)
        print(f"Fetching: {current_date}")
        logger.info(f"Fetching: {current_date}")

        repos.update(fetch_repositories(current_date.year, current_date.month, current_date.day, token))
        repos["saved_dates"].append(current_date)

        if current_date.day % 15 == 0:
            print("Saving data...")
            save_repos(repos, filename)
            logger.info(f"Number of repos gathered so far: {len(repos)-1}.")

        time.sleep(2)

    save_repos(repos, filename)
    logger.info(f"Total repos: {len(repos)}.")


if __name__ == "__main__":
    main()
