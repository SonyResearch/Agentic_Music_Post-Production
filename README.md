# Agentic Music Post-Production: project page

Static site for GitHub Pages. No build step.

To preview, run a server that supports range requests, such as `npx http-server`. Python's `http.server` doesn't, and switching mixes will restart the clip there. GitHub Pages works.

To publish, put these in the publishing root of the repository: `.nojekyll`, `index.html`, `site.css`, `site.js`, `data.js`, `peaks.js`, `critic.js`, `assets/`, `audio/`.

`index.html` holds the title, authors, links, and the overview video. Replace `assets/pipeline.png` to change the main figure, and keep the `width` and `height` attributes in step with the new image.

Noto Sans is licensed under the SIL Open Font License (`assets/fonts/OFL.txt`).
