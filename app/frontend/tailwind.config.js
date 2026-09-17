export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        mustard: {
          50: "#FBF6E8",
          100: "#F4E6C1",
          400: "#E2B33A",
          500: "#C9961A",
          600: "#A67912",
        },
        ink: {
          950: "#12110E",
          900: "#1B1914",
          800: "#26231C",
        },
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["Outfit", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        card: "0 18px 40px -24px rgba(24, 18, 8, 0.45)",
      },
    },
  },
  plugins: [],
};
