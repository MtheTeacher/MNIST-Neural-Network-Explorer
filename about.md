# About the MNIST Neural Network Explorer

... (previous sections) ...

## 14. Resolution Note: Comprehensive Infrastructure Overhaul for Cross-Device Reliability

### The Issue
The app was intermittently failing to load, particularly on mobile devices, with 404 errors for `index.tsx` and 404 errors for internal dependency chunks from the AI Studio CDN.

### Root Cause
1. **Absolute Paths**: The entry point was defined with a leading slash, which breaks when the app is not at the domain root.
2. **CDN Chunking**: The previous CDN provider had health issues serving sub-modules.
3. **Mobile Compatibility**: Lack of Import Map support in older mobile browsers.

### The Fix
1. **Relative Script Loading**: Changed `/index.tsx` to `index.tsx`.
2. **Standardized CDN**: Switched to `esm.sh` for all major dependencies to ensure consistent chunk delivery.
3. **Mobile Shim**: Integrated `es-module-shims` to polyfill Import Map support for students using older smartphones.
4. **Reliability Plan**: Created `RELIABILITY_PLAN.md` to document ongoing maintenance for cross-device support.

## 15. Resolution Note: Laptop-First Optimization & Bundle Cleanup

### The Goal
Prioritize reliability and performance for modern laptop environments, simplifying the initialization process.

### The Changes
1. **Shim Removal**: Removed `es-module-shims.js`. While useful for legacy mobile support, it adds a layer of complexity and potential boot-up lag that is unnecessary for modern desktop browsers.
2. **Path Hardening**: Reinforced relative pathing for the `index.tsx` entry point.
3. **Plan Update**: Formally established the `RELIABILITY_PLAN.md` as a living document to guide future infrastructure decisions.