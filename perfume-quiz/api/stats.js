import { head } from '@vercel/blob';

const BLOB_KEY = 'stats.json';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    res.status(405).json({ error: 'method_not_allowed' });
    return;
  }
  try {
    const info = await head(BLOB_KEY);
    const upstream = await fetch(info.url, { cache: 'no-store' });
    const data = upstream.ok ? await upstream.json() : {};
    res.setHeader('cache-control', 'no-store');
    res.status(200).json(data);
  } catch {
    res.setHeader('cache-control', 'no-store');
    res.status(200).json({});
  }
}
