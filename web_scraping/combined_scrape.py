import asyncio
import time

from httpx_async_demo import scrape_movies
from playwright.async_api import async_playwright


def write_to_file(data):
    with open("movie_info.txt", "a", encoding="utf-8") as file:
        file.write(data)
        file.write("\n")

def continue_movie_info():
    last_url = None
    lines_length = 0
    try:
        with open("movie_info.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
            lines_length = len(lines)
            last_url = lines[-1].split(";")[-1]
    except:
        print("No movie found")
    
    if last_url:
        return last_url + "reference", lines_length
    else:
        return last_url, lines_length

async def main():
    start_time = time.time()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.route("**/*", lambda route, request: asyncio.create_task(route.continue_()) if request.resource_type not in ['image'] else asyncio.create_task(route.abort()))

        await page.goto("https://www.imdb.com/search/title/?title_type=feature,tv_movie&sort=num_votes,desc", wait_until="domcontentloaded")

        last_url, last_line = continue_movie_info()
        if last_url:
            last_movie_page = await context.new_page()
            await last_movie_page.goto(last_url)
            total_votes = int(await last_movie_page.locator("span.ipl-rating-star__total-votes").inner_text().replace("(","").replace(")","").replace(".",""))
            await last_movie_page.close()

            await page.locator("div#numOfVotesAccordion").click()
            vote_input = await page.get_by_placeholder("e.g. 700000")
            await vote_input.fill(str(total_votes-1))

        for _ in range(3):
            for _ in range(2):
                for _ in range(3):
                    await page.click("button:has-text('50 more')")
                
                await page.wait_for_timeout(2000)
                titles = await page.locator("a > h3.ipc-title__text").all_inner_texts()
                urls = await page.locator("a.ipc-title-link-wrapper").evaluate_all("elements => elements.map(el => el.href)")
                
                print(titles)
                print(urls)

                for title, url in zip(titles, urls):
                    title = title[title.find(" ")+1:]
                    url = url[:(len(url) - url[::-1].find("/"))]
                    write_to_file(f"{title};{url}")
                
                last_url = urls[-1]
                last_url = last_url[:(len(last_url) - last_url[::-1].find("/"))]
                last_url = last_url + "reference"
                
                await page.locator("button.ipc-scroll-to-top-button.sc-87e28572-0.hJXYmZ.visible.ipc-chip.ipc-chip--on-base").click()

                detail_page = await context.new_page()
                await detail_page.goto(last_url)
                total_votes = int((await detail_page.locator("span.ipl-rating-star__total-votes").inner_text()).replace("(","").replace(")","").replace(".",""))
                await detail_page.close()

                await page.locator("div#numOfVotesAccordion").click()
                vote_input = page.get_by_placeholder("e.g. 700000")
                await vote_input.fill(str(total_votes-1))

            await scrape_movies(last_line)
            
        await context.storage_state(path="state.json")
        await context.close()
        await browser.close()
        
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

asyncio.run(main())
