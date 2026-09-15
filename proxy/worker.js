/**
 * A tiny proxy so the demo can answer in prose without publishing a key.
 *
 * An Anthropic key pasted into the page would be readable by anyone who views
 * source, and a key in a public repo is scraped and drained in minutes. The key
 * therefore lives HERE, as a Cloudflare secret the browser never sees, and the
 * page calls this instead of api.anthropic.com.
 *
 * Deploy (free tier, about five minutes):
 *
 *   npm install -g wrangler
 *   wrangler login
 *   cd proxy && wrangler deploy
 *   wrangler secret put ANTHROPIC_API_KEY      # paste the key when prompted
 *
 * Then put the printed URL into the page as PROXY_URL.
 *
 * Two guards, because a public proxy is a public endpoint:
 *   - ORIGIN allowlist, so only the demo's own page can call it.
 *   - a cap on how much can be asked at once, so a scraper cannot turn it into
 *     free inference. Neither makes it abuse-proof; they make it cheap to
 *     shut off. Rotate the key if the bill looks wrong.
 */

const ORIGINS = [
  'https://pranitmathur06.github.io',
  'http://127.0.0.1:8788',
];
const MAX_PROMPT_CHARS = 24000;
const MAX_TOKENS = 1200;

function cors(origin) {
  return {
    'Access-Control-Allow-Origin': ORIGINS.includes(origin) ? origin : ORIGINS[0],
    'Access-Control-Allow-Headers': 'content-type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Vary': 'Origin',
  };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: cors(origin) });
    }
    if (request.method !== 'POST') {
      return new Response('POST only', { status: 405, headers: cors(origin) });
    }
    if (!ORIGINS.includes(origin)) {
      return new Response('not allowed from this origin', { status: 403, headers: cors(origin) });
    }

    let body;
    try { body = await request.json() } catch (e) {
      return new Response('bad json', { status: 400, headers: cors(origin) });
    }
    const prompt = String(body.prompt || '');
    if (!prompt || prompt.length > MAX_PROMPT_CHARS) {
      return new Response('prompt missing or too long', { status: 400, headers: cors(origin) });
    }

    const upstream = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-sonnet-5',
        max_tokens: MAX_TOKENS,
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    // Pass the model's answer back, and nothing else -- never the key, never
    // the upstream error body, which can echo request details.
    if (!upstream.ok) {
      return new Response(JSON.stringify({ error: 'upstream ' + upstream.status }),
        { status: 502, headers: { ...cors(origin), 'content-type': 'application/json' } });
    }
    const data = await upstream.json();
    const text = (data.content || []).filter(c => c.type === 'text').map(c => c.text).join('');
    return new Response(JSON.stringify({ text }),
      { headers: { ...cors(origin), 'content-type': 'application/json' } });
  },
};
