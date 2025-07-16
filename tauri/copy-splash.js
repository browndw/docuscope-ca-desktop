#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('Copying splash screen files...');

// Create dist directory if it doesn't exist
const distDir = path.join(__dirname, '..', 'dist');
if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
}

// Copy splash screen files to dist
const files = [
    'src/splashscreen.html',
    'src/splash-styles.css'
];

files.forEach(file => {
    const src = path.join(__dirname, file);
    const dest = path.join(distDir, path.basename(file));
    
    try {
        fs.copyFileSync(src, dest);
        console.log(`Copied ${file} to ${dest}`);
    } catch (error) {
        console.error(`Error copying ${file}:`, error.message);
        process.exit(1);
    }
});

console.log('Splash screen files copied successfully!');
