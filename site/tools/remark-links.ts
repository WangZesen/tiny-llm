import { sourceLink } from '../src/lib/links.ts';
export default function remarkLinks() {
  return (tree: any, file: any) => {
    const absolute = String(file.path ?? file.history?.[0] ?? '').replaceAll('\\', '/');
    const start = absolute.lastIndexOf('/doc/');
    if (start < 0) return;
    const source = absolute.slice(start + 1);
    const visit = (node: any) => {
      if (['link', 'image', 'definition'].includes(node.type) && node.url)
        node.url = sourceLink(source, node.url);
      if (node.children) node.children.forEach(visit);
    };
    visit(tree);
    // Layout supplies the only page-level heading.
    tree.children = tree.children.filter(
      (node: any) => node.type !== 'heading' || node.depth !== 1,
    );
  };
}
