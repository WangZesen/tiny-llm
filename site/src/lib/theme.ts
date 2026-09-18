import { useEffect, useState } from 'react';

/** Tracks the theme Layout.astro stamps on <html> and announces with `themechange`. */
export function useTheme() {
  const [theme, setTheme] = useState('light');
  useEffect(() => {
    const update = () => setTheme(document.documentElement.dataset.theme ?? 'light');
    update();
    window.addEventListener('themechange', update);
    return () => window.removeEventListener('themechange', update);
  }, []);
  return theme;
}
