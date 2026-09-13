import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { load } from 'cheerio';
const directory = path.resolve('dist');
const walk = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((f) =>
    f.isDirectory() ? walk(path.join(dir, f.name)) : [path.join(dir, f.name)],
  );
const pages = walk(directory).filter((p) => p.endsWith('.html'));
const errors: string[] = [];
for (const file of pages) {
  const relative = path.relative(directory, file).replaceAll('\\', '/');
  const current = new URL(
    '/tiny-llm/' + relative.replace(/index\.html$/, ''),
    'https://local.invalid',
  );
  const $ = load(readFileSync(file, 'utf8'));
  if ($('.katex-error').length) errors.push(`${relative}: KaTeX rendering error`);
  for (const node of $('a[href], img[src], script[src], link[href]').toArray()) {
    const source = $(node).attr('href') ?? $(node).attr('src')!;
    const target = new URL(source, current);
    if (target.origin !== current.origin) continue;
    if (!target.pathname.startsWith('/tiny-llm/')) {
      errors.push(`${relative}: URL escapes base: ${source}`);
      continue;
    }
    let dest = path.join(directory, decodeURIComponent(target.pathname.slice('/tiny-llm/'.length)));
    if (existsSync(dest) && statSync(dest).isDirectory()) dest = path.join(dest, 'index.html');
    if (!existsSync(dest)) {
      errors.push(`${relative}: missing ${source}`);
      continue;
    }
    if (target.hash && dest.endsWith('.html')) {
      const ids = new Set(
        load(readFileSync(dest, 'utf8'))('[id]')
          .map((_, el) => load(el).root().children().attr('id'))
          .get(),
      );
      if (!ids.has(decodeURIComponent(target.hash.slice(1))))
        errors.push(`${relative}: missing anchor ${source}`);
    }
  }
}
if (errors.length) {
  console.error(errors.join('\n'));
  process.exit(1);
}
console.log(`Checked links, assets, anchors, and math in ${pages.length} HTML pages.`);
