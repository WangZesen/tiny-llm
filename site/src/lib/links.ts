import path from 'node:path';
export const base = '/tiny-llm/';
export const url = (p = '') => base + p.replace(/^\//, '');
export function sourceLink(source: string, target: string): string {
  if (/^(?:[a-z]+:|#|\/)/i.test(target)) return target;
  const [file, anchor] = target.split('#');
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(source), file));
  const suffix = anchor ? '#' + anchor : '';
  if (resolved.startsWith('doc/data/')) return url('assets/' + resolved.slice(4)) + suffix;
  if (resolved.startsWith('doc/adaptive-consensus-investigation/'))
    return url('assets/' + resolved.slice(4)) + suffix;
  if (resolved.startsWith('configs/')) return url('assets/' + resolved) + suffix;
  if (resolved.startsWith('doc/') && resolved.endsWith('.md'))
    return url(resolved.slice(4, -3).replace(/\/index$/, '') + '/') + suffix;
  return 'https://github.com/WangZesen/tiny-llm/blob/main/' + resolved + suffix;
}
