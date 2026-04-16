import { head, put } from '@vercel/blob';

const BLOB_KEY = 'stats.json';
const SLUG_RE = /^[a-z0-9-]+$/;

async function readStats() {
  try {
    const info = await head(BLOB_KEY);
    const res = await fetch(info.url, { cache: 'no-store' });
    if (!res.ok) return {};
    return await res.json();
  } catch {
    return {};
  }
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method_not_allowed' });
    return;
  }

  let body = req.body;
  if (typeof body === 'string') {
    try {
      body = JSON.parse(body);
    } catch {
      res.status(400).json({ error: 'invalid_json' });
      return;
    }
  }
  const slug = body?.slug;
  if (typeof slug !== 'string' || !SLUG_RE.test(slug) || slug.length > 80) {
    res.status(400).json({ error: 'invalid_slug' });
    return;
  }

  const stats = await readStats();
  stats[slug] = (stats[slug] || 0) + 1;

  await put(BLOB_KEY, JSON.stringify(stats), {
    access: 'public',
    contentType: 'application/json',
    allowOverwrite: true,
    addRandomSuffix: false,
  });

  res.status(200).json({ ok: true, count: stats[slug] });
}
