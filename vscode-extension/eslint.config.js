module.exports = [
  {
    ignores: ["node_modules/**", "*.vsix"],
    files: ["**/*.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "commonjs",
    },
    rules: {},
  },
];
