const esbuild = require('esbuild');
const { glob } = require('glob');
const path = require('path');

async function build() {
  const entryPoints = await glob('src/**/*.ts');
  
  // Get all dependencies from package.json to mark as external
  const pkg = require('./package.json');
  const allDeps = Object.keys(pkg.dependencies || {});
  
  await esbuild.build({
    entryPoints,
    platform: 'node',
    target: 'node18',
    outdir: 'dist',
    format: 'cjs',
    bundle: false,
    outbase: 'src',
    // Ensure proper module resolution
    mainFields: ['main', 'module'],
    resolveExtensions: ['.ts', '.js', '.json'],
  });
  
  console.log('Build complete!');
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});

