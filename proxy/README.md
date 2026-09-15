# The ask proxy

The demo answers from its own rows with no model at all. This adds prose
answers for everyone, without publishing a key.

A key in the page would be readable in view-source and scraped within minutes,
so it lives as a Cloudflare secret instead and the page calls this Worker.

```bash
npm install -g wrangler
wrangler login
cd proxy
wrangler deploy
wrangler secret put ANTHROPIC_API_KEY     # paste the key at the prompt
```

`wrangler deploy` prints a URL like `https://courtvision-ask.<you>.workers.dev`.
Put it in `PROXY_URL` at the top of the page script and rebuild.

Free tier covers 100k requests a day. If the bill ever looks wrong, rotate the
key — the page keeps working, it just falls back to answering from the rows.
