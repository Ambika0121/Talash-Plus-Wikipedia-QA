from urllib.parse import urlparse, unquote
import wikipediaapi

def scrape_wikipedia_pages(urls_input: str):
    """Fetch text from comma-separated English Wikipedia article URLs."""
    urls = [u.strip() for u in urls_input.split(",") if u.strip()]
    wiki = wikipediaapi.Wikipedia(
        user_agent="TalashPlusWikipediaQA/1.0 (educational project)", language="en"
    )
    pages = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc.lower() not in {"en.wikipedia.org", "www.wikipedia.org"}:
            raise ValueError(f"Only English Wikipedia URLs are supported: {url}")
        title = unquote(parsed.path.rsplit("/", 1)[-1]).replace("_", " ")
        if not title:
            continue
        page = wiki.page(title)
        if page.exists() and page.text.strip():
            pages.append({"title": page.title, "url": url, "text": page.text.strip()})
    return pages
