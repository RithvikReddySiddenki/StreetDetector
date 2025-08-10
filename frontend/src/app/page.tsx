// src/app/page.tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import { MODEL_URL } from "@/lib/modelConfig";
import { getOnnxSession } from "@/lib/onnxSession";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);  // shows preview then gets replaced by result
  const [banner, setBanner] = useState<"Preview" | "Detections" | null>(null);
  const [loading, setLoading] = useState(false);
  const [sessionReady, setSessionReady] = useState(false);

  // Keep a stable URL value
  const modelUrl = useMemo(() => MODEL_URL, []);

  useEffect(() => {
    // Load the model once on the client
    let mounted = true;
    (async () => {
      try {
        await getOnnxSession(modelUrl);
        if (mounted) setSessionReady(true);
      } catch (e) {
        console.error("Failed to load ONNX model:", e);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [modelUrl]);

  function onSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setBanner("Preview");
    setImageUrl(URL.createObjectURL(f));
  }

  async function onDetect() {
    if (!file || !sessionReady) return;
    setLoading(true);

    try {
      // 1) Read the image as ImageBitmap or HTMLImageElement
      const img = await readFileAsImage(file);

      // 2) TODO: preprocess to the input tensor shape (e.g., 640x640 NCHW float32)
      // const { inputTensor, letterboxMeta } = await preprocessToTensor(img);

      // 3) Run session
      const session = await getOnnxSession(modelUrl);
      // Example assuming the input name is "images" (check your model):
      // const outputs = await session.run({ images: inputTensor });

      // 4) TODO: decode YOLO outputs to boxes, draw on a canvas
      // const drawnBlob = await drawDetectionsOnImage(img, outputs, letterboxMeta);

      // For now, just keep the same preview to prove the flow works.
      // Replace with the blob you draw:
      // const url = URL.createObjectURL(drawnBlob);
      // setImageUrl(url);

      setBanner("Detections");
      // Remove the line below once you set the real result image:
      // setImageUrl(url);
    } catch (err) {
      console.error(err);
      alert("Detection failed. See console for details.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-white">
      <div className="mx-auto w-full max-w-screen-xl px-6 py-10 space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold">StreetDetector</h1>
          <p className="text-neutral-300">
            The model is served from {modelUrl}. No backend needed on Vercel.
          </p>
        </header>

        <div className="flex items-center gap-4">
          <label className="inline-flex cursor-pointer rounded-xl bg-neutral-800 hover:bg-neutral-700 px-4 py-2">
            <input type="file" accept="image/*" className="hidden" onChange={onSelect} />
            Choose Image
          </label>

          <button
            onClick={onDetect}
            disabled={!file || loading || !sessionReady}
            className="rounded-xl bg-blue-600 disabled:bg-blue-900 hover:bg-blue-500 px-4 py-2"
          >
            {loading ? "Detecting..." : sessionReady ? "Detect" : "Loading Model..."}
          </button>
        </div>

        {imageUrl && (
          <section className="space-y-3">
            <h2 className="text-xl font-medium">{banner}</h2>
            <div className="rounded-2xl bg-neutral-900 p-4 shadow-lg">
              <img
                src={imageUrl}
                alt="current"
                className="w-full h-auto max-w-[1280px] mx-auto rounded-xl"
              />
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

// Utilities

function readFileAsImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}
