# Project page deployment

## Public locations

| Content | URL |
| --- | --- |
| Project page | https://internrobotics.github.io/RoboAtom/ |
| Code | https://github.com/InternRobotics/RoboAtom |
| Data, benchmark, and media | https://huggingface.co/datasets/InternRobotics/RoboAtom |
| Models | https://huggingface.co/InternRobotics/RoboAtom |
| Paper | https://arxiv.org/abs/2610.40207 |

Images and videos load from `project_media/` in the Hugging Face dataset repository: figures in `figures/`, posters in `posters/`, logos in `brand/`, and videos in `videos/overview/`, `videos/demos/`, and `videos/wall/`. GitHub Pages serves HTML, CSS, JavaScript, the icon sprite, and the README badge assets.

The README resource rows use standalone SVG images in `assets/badges/`, based on the same icons as the website. Their public image URLs point to the code repository's raw `main` files so they work on GitHub and Hugging Face without a badge service. Keep the inline display attributes on both the links and images: Hugging Face prose styles otherwise stack badge links vertically. GitHub can use its native inline image rendering. Publish these assets before updating the Hugging Face README cards; until the code changes are pushed, their new public URLs will not exist.

The Paper link is set in `index.html` (two `data-resource="paper"` links) and in `RESOURCE_LINKS.paper` in `script.js`.

The final section before the footer contains the paper's BibTeX citation and a copy button. The citation remains readable without JavaScript; if clipboard access is unavailable or denied, users can select and copy the text manually. Keep it consistent with the three release READMEs and the paper metadata in `CITATION.cff`.

## Build

From `homepage/`:

```bash
npm run build:pages
```

The build uses only Node.js built-in modules and writes `dist/RoboAtom/`.

## Figure layout

The three release READMEs center the overview image, cap it at 1040px, and link to the full-resolution original. Keep both the HTML width attribute (for GitHub) and the responsive inline styles (for HF and wide Markdown previews).

Website construction and control figures are centered at maximum widths of 1120px and 1040px. Data composition and training progress retain their 900px cap. The desktop benchmark columns use a 1:1.61 width ratio so the two original images have the same displayed height; at 900px and below they stack vertically. All figures retain automatic height and their original aspect ratio, with the existing image viewer available for details.

## GitHub Pages

In Settings → Pages, select GitHub Actions. The `deploy-pages.yml` workflow builds `homepage/` on pushes to `main` and publishes `homepage/dist/RoboAtom/` as the site root. Upload `project_media/` to Hugging Face before publishing the page.

## Browser check

```bash
npm install
npm run build:pages
npm run test:citation
npm run test:links
BASE_URL="http://127.0.0.1:3000/RoboAtom/" npm run test:browser
BASE_URL="https://internrobotics.github.io/RoboAtom/" npm run test:browser
```

`test:citation` checks the built page at the site root and under `/RoboAtom/`, desktop and mobile layouts, Paper links, citation content, clipboard success and failure, keyboard access, and the no-JavaScript fallback. It serves local fixtures through browser request interception and does not download public media. Run `npm run build:pages` before this focused check.

`test:links` checks the built page's link destinations, icons, hover and keyboard focus styles, mobile navigation, README badge rendering, and popup behavior without accessing the network. These local checks do not replace checking the public badge URLs after publication.
