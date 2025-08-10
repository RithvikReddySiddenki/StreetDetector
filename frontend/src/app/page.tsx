"use client";
import { useState } from "react";
import DetectModal from "@/components/DetectModal";

export default function Home() {
  const [open, setOpen] = useState(false);

  // Example: call your backend (/api/detect) or run onnxruntime-web here.
  async function onDetect(file: File): Promise<Blob | string> {
    // Backend example returning an image Blob:
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/detect", { method: "POST", body: form });
    if (!res.ok) throw new Error(`Backend error: ${res.status}`);
    return await res.blob(); // the modal will display this as an image
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-white p-8">
      <h1 className="text-3xl font-semibold">StreetDetector</h1>
      <p className="text-neutral-300 mt-2">
        Upload an image and run detection.
      </p>

      <button
        onClick={() => setOpen(true)}
        className="mt-6 rounded-xl bg-blue-600 px-4 py-2 hover:bg-blue-500"
      >
        Open detector
      </button>

      <DetectModal open={open} onClose={() => setOpen(false)} onDetect={onDetect} />
    </main>
  );
}
