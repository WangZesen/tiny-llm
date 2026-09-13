import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const articles = defineCollection({
  loader: glob({ base: '../doc', pattern: '{results,methods,guides,performance,archive}/*.md' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    archive: z.boolean().default(false),
  }),
});
export const collections = { articles };
