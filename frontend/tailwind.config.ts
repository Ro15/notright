
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#0f766e",
        secondary: "#0ea5e9",
      },
    },
  },
  plugins: [],
};

export default config;
