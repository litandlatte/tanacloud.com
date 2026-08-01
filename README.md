# tanacloud.com

Rohit Kumar's data science site — talks, workshops, notebooks and notes.
Plain static HTML/CSS/JS. No build step, no framework, no external requests.

This folder maps 1:1 to Hostinger `public_html`.

## Short links

`tanacloud.com/h3` → the H3 workshop slides. Each short link is implemented **twice**:

1. a `301` in `.htaccess` (the real redirect), and
2. a fallback `<slug>/index.html` with meta-refresh + JS + canonical.

The fallback exists because **Hostinger's ZIP extractor silently skips dotfiles**, so
`.htaccess` has a history of not landing on deploy. Create it by hand in File Manager
every time, and verify by behaviour — a `403` on `/.htaccess` proves nothing on LiteSpeed.

To add one: add a `RewriteRule` line, copy `h3/index.html` and swap the destination URLs,
then add the entry to `hub.html`.

## Local preview

Use **Apache**, not `python3 -m http.server` — the latter ignores `.htaccess` and therefore
cannot show the redirects. See the project `CLAUDE.md` for the working macOS recipe.
