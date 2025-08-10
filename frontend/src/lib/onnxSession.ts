let _sessionPromise:
  | Promise<import("onnxruntime-web").InferenceSession>
  | null = null;

export async function getOnnxSession(modelUrl: string) {
  // Never run on the server
  if (typeof window === "undefined") {
    throw new Error("getOnnxSession must be called on the client.");
  }

  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    // Dynamic import so bundlers keep it client-only
    const ort: typeof import("onnxruntime-web") = await import("onnxruntime-web");

    // Optional: point to a known WASM CDN (often fixes 404s or MIME issues)
    // ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/";

    // Optional tweaks
    // ort.env.wasm.numThreads = 1;
    // ort.env.wasm.proxy = true;

    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return session;
  })();

  return _sessionPromise;
}
