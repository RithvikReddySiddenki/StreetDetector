let _sessionPromise: Promise<import("onnxruntime-web").InferenceSession> | null = null;

export async function getOnnxSession(modelUrl: string) {
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    const ort = await import("onnxruntime-web"); // dynamic to avoid SSR
    // Defaults are fine; you can tweak if needed:
    // ort.env.wasm.numThreads = 1;

    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return session;
  })();

  return _sessionPromise;
}
