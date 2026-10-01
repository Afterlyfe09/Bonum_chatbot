# Bonum AI Assistant

A website-specific customer-support chatbot for **Bonum eDesigns LLP** (https://bonum.in/).
It answers only from content crawled from the official website, shows the pages it used as
clickable sources, and says so plainly when something (such as a price) is not on the site.

## Features
- RAG pipeline: crawl -> clean -> chunk -> embed -> FAISS -> retrieve -> grounded LLM answer
- Relevance threshold: weak retrieval never reaches the LLM (prevents hallucination)
- Light hybrid retrieval: semantic search plus small keyword / page-intent boosts
- Strict system prompt: no invented prices, features, timelines, contacts or policies
- Out-of-scope handling, prompt-injection / secret-leak refusal
- Follow-up questions ("tell me more about the first one") via query rewriting; history never overrides the site
- Source citations built only from retrieved URLs
- Friendly handling of: missing key, timeouts, rate limits, network errors, missing/corrupt index, empty input
- Persistent index: embeddings are built once by `ingest.py`; the app starts instantly

## Architecture
```
bonum.in -> scraper.py -> cleaner.py -> chunker.py -> embeddings.py -> FAISS (vectorstore/)
                                                                          |
user question -> chatbot.py (guardrails, follow-up rewrite) -> retriever.py (top-k + threshold)
              -> LLM with only the retrieved excerpts -> answer + sources -> app.py (Streamlit)
```
| File | Responsibility |
|---|---|
| `ingest.py` | Runs the whole ingestion pipeline |
| `src/scraper.py` | Crawls bonum.in only (same domain, `.php` pages, blog posts); retries; raw HTML cache |
| `src/cleaner.py` | Strips scripts/forms, keeps headings, removes filler text, separates repeated header/footer text into one "Contact and site information" document |
| `src/chunker.py` | ~900-character chunks that keep page title + section heading |
| `src/embeddings.py` | `all-MiniLM-L6-v2` (local, free) |
| `src/retriever.py` | FAISS cosine search, boosts, relevance threshold |
| `src/chatbot.py` | Guardrails, rewrite, prompt assembly, OpenAI call, error mapping |
| `src/prompts.py` | System prompt and fixed messages |
| `app.py` | Streamlit UI |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your OPENAI_API_KEY in .env
python ingest.py
streamlit run app.py
```
Environment variables (`.env`): `OPENAI_API_KEY` (required); optional `OPENAI_MODEL` (default
`gpt-4o-mini`), `OPENAI_BASE_URL` (any OpenAI-compatible provider), `RELEVANCE_THRESHOLD`
(default `0.30`), `TOP_K` (default `5`).

## Updating the knowledge base
Run `python ingest.py` again; it overwrites `vectorstore/`. Restart Streamlit afterwards.

If a page cannot be downloaded (some Bonum pages answered HTTP 409 to automated requests during
development), the crawler retries and then falls back to `data/raw/<name>.html`. Open the page in
your browser, "Save page as" HTML, and save it using the cache name, e.g. `data/raw/faq_php.html`,
`data/raw/contact_php.html`, `data/raw/about_php.html`. `python ingest.py --offline` rebuilds from
that folder only. The ingest summary lists every page and its extracted line count so you can
verify nothing important is missing.

## How RAG works here
1. The website is split into chunks, each turned into a vector that captures its meaning.
2. A question is turned into a vector and compared with all chunks, so "What can Bonum help my business with?" finds the services text even without shared keywords.
3. Chunks below the relevance threshold are discarded; if none remain, the bot says it has nothing on that topic.
4. The remaining chunks are given to the LLM with instructions to answer only from them. If the answer is not there, it must say so.
5. The pages those chunks came from are shown as sources.

## Testing
```bash
pytest -q                           # offline unit tests (no API key, no index needed)
python -m tests.test_chatbot --live # 22 questions against the real index and LLM
```
The live run prints status, retrieval score, answer and sources for each question. Use the scores
to tune `RELEVANCE_THRESHOLD`: unrelated questions (cricket, Python code) should fall below it,
real service questions above it. Check pricing/contact answers by hand against bonum.in.

## Example questions
What does Bonum do? / Tell me about web development. / Compare web development and app development. /
How can I contact Bonum? / Where is Bonum located? / How much does web development cost? /
What projects has Bonum worked on? / What is your refund policy?

## Deployment (Streamlit Community Cloud)
1. Run `python ingest.py` locally and commit the `vectorstore/` folder (`.env` stays ignored).
2. Push to GitHub, create an app on https://share.streamlit.io pointing at `app.py`.
3. In the app's Secrets add: `OPENAI_API_KEY = "sk-..."`.
The embedding model downloads on the first query, so the first answer after a cold start is slower.

## Limitations
- Knows only what was on the site at the last ingestion; re-run `ingest.py` after site changes.
- Text only: information inside images (such as the portfolio screenshots) is not indexed.
- Answer quality depends on the site's own text; very short pages give short answers.
- Index uses pure-Python JSON metadata and a flat FAISS index, which is ideal for a site this size but not for millions of chunks.
