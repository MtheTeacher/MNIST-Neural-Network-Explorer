# MNIST Data Source Guidelines

This document contains recommendations for sourcing the MNIST dataset for in-browser applications, ensuring reliability and avoiding common CORS or authentication issues. This is based on expert advice and should be consulted before changing data source URLs in `services/mnistData.ts`.

## Summary

Use a Google-hosted mirror for dead-simple, CORS-friendly browser fetches. Avoid Kaggle for in-browser fetching due to authentication and CORS complications. The primary goal is to use rock-solid sources that work well in a browser environment.

---

### 1. Google’s TFJS Sprite + Labels (Primary Choice)

This is the ideal source for our application's current sprite-sheet extraction pipeline. These are the exact files used by the official TensorFlow.js examples, hosted on a public Google Cloud Storage (GCS) bucket with permissive CORS settings.

-   **Images:** `https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png`
-   **Labels:** `https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8`

**Why it's the best fit:**
-   **Proven Reliability:** Used in official TFJS demos.
-   **CORS Friendly:** Public GCS buckets are configured for anonymous, cross-origin access.
-   **Perfect Match:** Directly compatible with our `<canvas>`-based sprite parsing logic.

---

### 2. Alternative Data Formats (Fallback or Future Use)

If the sprite sheet approach were to be replaced, these are other reliable sources.

#### CVDF Mirror of Canonical IDX Files

The Computer Vision Datasets Foundation hosts a mirror of the original `ubyte` files on GCS. This would require implementing a client-side IDX parser.

-   `https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz`
-   `https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz`
-   `https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz`
-   `https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz`

#### Single-File NPZ from TensorFlow/Keras

Keras provides the dataset as a single `mnist.npz` file, also on GCS. This is very compact but would require a small, client-side NPZ parser library.

-   `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`

#### Hugging Face Datasets

Hugging Face hosts an official mirror with a high-availability CDN and a dataset-server API. This is powerful but more complex than a direct file fetch.

-   **Dataset Hub Page:** [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)

---

### What to Avoid

-   **Canonical Source (yann.lecun.org):** While it's the official source of truth, it's not designed for high-availability direct fetching from web apps and can have access issues. Use it for documentation and checksums, not as a production origin.
-   **Kaggle:** Requires authentication and has strict CORS policies, making it unsuitable for direct, anonymous fetching from a browser.

### Recommended Strategy for This App

1.  **Primary Source:** Use the TFJS sprite and labels from the `learnjs-data/model-builder` GCS bucket. This is the cleanest, most direct solution that matches our current architecture.
2.  **Fallbacks:** Maintain a list of other reliable sprite-sheet sources (like the other GCS mirrors) in `services/mnistData.ts` to automatically handle temporary outages.
3.  **Future-Proofing:** If a major change is needed, consider switching the parsing logic to handle the Keras `mnist.npz` file as the next most efficient alternative.
