import asyncio
import json
import time
from asyncio import Semaphore

import httpx


# Read movie URLs from file
def read_movie_info(start_line = 0):
    with open("movie_info.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()[start_line:]
        urls = [line.split(";")[-1].strip() for line in lines]
    return urls

def write_detail_movie_info(data):
    with open("detail_movie_info_async.txt", "a", encoding="utf-8") as file:
        file.write(data)
        file.write("\n")

async def fetch_movie_data(client, url, headers, semaphore):
    async with semaphore:
        try:
            response = await client.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()
        except httpx.RequestError as e:
            print(f"Error navigating to {url}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"HTTP error on {url}: {e.response.status_code}")
            return None

        json_data = None
        start_index = response.text.find('<script type="application/ld+json">')
        if start_index != -1:
            end_index = response.text.find('</script>', start_index)
            if end_index != -1:
                json_data = response.text[start_index + len('<script type="application/ld+json">'):end_index].strip()

        if json_data:
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for {url}: {e}")
                return None

            movie_data = {
                "name": data.get("name"),
                "turkish_name": data.get("alternateName"),
                "type": data.get("@type"),
                "poster_url": data.get("image"),
                "description": data.get("description"),
                "review_name": data.get("review", {}).get("name"),
                "review_body": data.get("review", {}).get("reviewBody"),
                "nov": data.get("aggregateRating", {}).get("ratingCount"),
                "rating": data.get("aggregateRating", {}).get("ratingValue"),
                "age_limit": data.get("contentRating"),
                "genre": data.get("genre"),
                "published_date": data.get("datePublished"),
                "keywords": data.get("keywords"),
                "actors": {actor.get("name") for actor in data.get("actor", [])},
                "actors_url": {actor.get("url") for actor in data.get("actor", [])},
                "directors": {director.get("name") for director in data.get("director", [])},
                "writers": {creator.get("name") for creator in data.get("creator", [])},
                "writer_urls": {creator.get("url") for creator in data.get("creator", []) if creator.get("@type") == "Person"},
                "company_urls": {creator.get("url") for creator in data.get("creator", []) if creator.get("@type") == "Organization"},
                "duration": data.get("duration")
            }


            if not movie_data["company_urls"]:
                movie_data["company_urls"] = "None"

            formatted_data = f"""{movie_data['name']}~{movie_data['turkish_name']}~{movie_data['type']}~{movie_data['poster_url']}~{movie_data['description']}~
                {movie_data['review_name']}~{movie_data['review_body']}~{movie_data['nov']}~{movie_data['rating']}~{movie_data['age_limit']}~{movie_data['genre']}~
                {movie_data['published_date']}~{movie_data['keywords']}~{movie_data['actors']}~{movie_data['actors_url']}~{movie_data['directors']}~{movie_data['writers']}~
                {movie_data['writer_urls']}~{movie_data['company_urls']}~{movie_data['duration']}"""
            formatted_data = formatted_data.replace("\r","").replace("\n","").replace(";",",").replace("~",";").replace("&apos,","'").replace("&quot,","\"")
            write_detail_movie_info(formatted_data)

# Asynchronous main function to scrape all movie data
async def scrape_movies(start_line = 0):
    urls = read_movie_info(start_line)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    }
    
    semaphore = Semaphore(9)  # Limit concurrent requests
    async with httpx.AsyncClient() as client:
        tasks = [fetch_movie_data(client, url, headers, semaphore) for url in urls]
        await asyncio.gather(*tasks)

start_time = time.time()
asyncio.run(scrape_movies())
print(f"Scraping finished in {time.time() - start_time} seconds")