# MNIST Data Source Decision Record

This document records the decision to change the primary source for the MNIST dataset.

## Status: Active

### Current Data Source: Embedded Data Pointers

The application no longer fetches the MNIST dataset from a hardcoded URL that could change or become unreliable. Instead, the locations of the stable data files are **embedded as constants** in `services/embeddedData.ts`. These constants point to the official, reliable Google Cloud Storage URLs:
- `https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png`
- `https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8`

### Rationale

The application has previously used both different Google Cloud Storage URLs and the jsDelivr CDN. Both sources proved to be intermittently unreliable, leading to `Failed to fetch` errors for users.

By embedding the pointers to the canonical, most stable URLs, we achieve several goals:
- **Reliability:** We are using the most stable, long-term source for the data, as used by official TensorFlow.js examples. This mitigates the risk of data loading failures that have been a recurring issue.
- **Maintainability:** If the data source ever needs to be updated, it can be changed in a single, dedicated file (`services/embeddedData.ts`).

This approach was chosen over fully embedding the dataset as multi-megabyte Base64 strings, which would significantly increase the application's initial load time and bundle size. Using a reliable external source provides the best balance of reliability and performance. This decision represents the most robust solution to the recurring data loading problem.
