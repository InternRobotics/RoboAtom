import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputRoot = path.join(projectRoot, 'dist', 'RoboAtom');

// Copy only public runtime files; source, tests, and local configuration stay out.
await rm(outputRoot, { recursive: true, force: true });
await mkdir(outputRoot, { recursive: true });
for (const name of ['index.html', 'styles.css', 'script.js', 'assets']) {
  await cp(path.join(projectRoot, name), path.join(outputRoot, name), {
    recursive: true,
    filter: (source) => !source.toLowerCase().endsWith('.mp4'),
  });
}
await writeFile(path.join(outputRoot, '.nojekyll'), '');

const page = await readFile(path.join(outputRoot, 'index.html'), 'utf8');
if (/\b(?:src|href|poster)=["']\/(?!\/)/.test(page)) {
  throw new Error('Published assets must support the /RoboAtom/ subpath.');
}
console.info('GitHub Pages files prepared in dist/RoboAtom/');
