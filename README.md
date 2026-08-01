# tanacloud.com

Rohit Kumar's data science site — talks, workshops, notebooks and notes.
Plain static HTML/CSS/JS. No build step, no framework, no external requests.

## Hosting

Deployed on **Cloudflare Pages**, connected to this repo. Pushing to `main` deploys.
`DEV` is the working branch; merge `DEV` → `main` to release.

`.htaccess` is kept in the repo only so the site stays portable to Apache/LiteSpeed
hosting (Hostinger). **Cloudflare Pages ignores it** — it reads `_redirects` and `_headers`.

## Short links

`tanacloud.com/h3` → the H3 workshop slides.

**To add a short link, add one line to `_redirects`:**

```
/<slug>   <destination URL>   301
```

That is the whole job. Then optionally add the entry to `hub.html` so it is listed.

⚠️ **Do not also create a `<slug>/index.html` page.** On Cloudflare Pages a real static
file takes precedence over a redirect rule, so the page would shadow the `301` and the
short link would stop redirecting cleanly. (The old `h3/index.html` fallback existed only
to survive Hostinger's ZIP extractor dropping dotfiles — `_redirects` is a normal filename,
so that risk is gone.)

## Local preview

`python3 -m http.server` is fine for checking pages, but it does **not** apply `_redirects`,
so it cannot show the short links working. To test redirects, use `npx wrangler pages dev .`
or just verify on the Cloudflare preview deployment.
