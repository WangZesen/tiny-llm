import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkLinks from './tools/remark-links.ts';
import { unified } from '@astrojs/markdown-remark';

export default defineConfig({
  site: 'https://wangzesen.github.io',
  base: '/tiny-llm',
  trailingSlash: 'always',
  integrations: [react()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkLinks],
      rehypePlugins: [[rehypeKatex, { strict: 'error', throwOnError: true }]],
    }),
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
  },
});
