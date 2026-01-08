# Reliability Plan: Desktop & Laptop Focus

This plan outlines the infrastructure decisions made to ensure the MNIST Explorer remains stable for the primary target audience: students using modern laptops.

## 1. Pathing Strategy
- **Issue**: Absolute paths (e.g., `/index.tsx`) often result in 404 errors if the app is served from a subpath or a non-root environment.
- **Solution**: Use relative paths (e.g., `index.tsx`) for all internal resource imports. This makes the application root-agnostic.

## 2. Dependency Management (The "Two Reacts" Bug)
- **Issue**: `Uncaught TypeError: Cannot read properties of null (reading 'useRef')`. This is caused by multiple versions of React being loaded into the browser simultaneously.
- **Solution**: 
    - Forced `recharts` to treat React as an external dependency via `?external=react,react-dom` in the `esm.sh` URL.
    - Explicitly mapped `react-dom/client` in the `importmap` to sync the main entry point with the library's internal requirements.
    - Centralized all imports on `esm.sh` for robust, standards-compliant ESM bundling.

## 3. Platform Scoping
- **Decision**: Currently prioritizing "Modern Laptops".
- **Action**: Removed `es-module-shims.js`. This reduces overhead and potential race conditions during the initial boot sequence. Modern browsers handle `<script type="importmap">` natively.

## 4. Troubleshooting Checklist
- **If 404 persists**: Verify the server is configured to serve `.tsx` files as `application/javascript` or use a build tool like Vite to handle the transformation.
- **If TFJS fails**: Ensure hardware acceleration (WebGL) is enabled in the browser settings, as TensorFlow.js relies on the GPU for high-performance training.