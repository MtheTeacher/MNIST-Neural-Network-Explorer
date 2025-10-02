import type { Config } from 'tailwindcss';
import typography from '@tailwindcss/typography';

const config: Config = {
  content: [
    './index.html',
    './App.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './services/**/*.{ts,tsx}',
    './constants.tsx',
    './types.ts',
  ],
  theme: {
    extend: {},
  },
  plugins: [typography],
};

export default config;
