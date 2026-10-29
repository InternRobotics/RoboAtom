# Project page deployment

## Public locations

| Content | URL |
| --- | --- |
| Project page | https://internrobotics.github.io/RoboAtom/ |
| Code | https://github.com/InternRobotics/RoboAtom |
| Data, benchmark, and media | https://huggingface.co/datasets/InternRobotics/RoboAtom |
| Models | https://huggingface.co/InternRobotics/RoboAtom |
| Paper | https://arxiv.org/abs/2610.40207 |

Images and videos load from `project_media/` in the Hugging Face dataset repository: figures in `figures/`, posters in `posters/`, logos in `brand/`, and videos in `videos/overview/`, `videos/demos/`, and `videos/wall/`. GitHub Pages serves only HTML, CSS, JavaScript, and the icon sprite.

The Paper link is set in `index.html` (two `data-resource="paper"` links) and in `RESOURCE_LINKS.paper` in `script.js`.

The final section before the footer contains the paper's BibTeX citation and a copy button. The citation remains readable without JavaScript; if clipboard access is unavailable or denied, users can select and copy the text manually. Keep it consistent with the three release READMEs and the paper metadata in `CITATION.cff`.

## Build

From `homepage/`:

```bash
npm run build:pages
```

The build uses only Node.js built-in modules and writes `dist/RoboAtom/`.

## GitHub Pages

In Settings → Pages, select GitHub Actions. The `deploy-pages.yml` workflow builds `homepage/` on pushes to `main` and publishes `homepage/dist/RoboAtom/` as the site root. Upload `project_media/` to Hugging Face before publishing the page.

## Browser check

```bash
npm install
npm run build:pages
npm run test:citation
BASE_URL="http://127.0.0.1:3000/RoboAtom/" npm run test:browser
BASE_URL="https://internrobotics.github.io/RoboAtom/" npm run test:browser
```

`test:citation` checks the built page at the site root and under `/RoboAtom/`, desktop and mobile layouts, Paper links, citation content, clipboard success and failure, keyboard access, and the no-JavaScript fallback. It serves local fixtures through browser request interception and does not download public media. Run `npm run build:pages` before this focused check.
