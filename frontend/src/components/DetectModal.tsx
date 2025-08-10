"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

type DetectModalProps = {
  /** Show / hide the modal */
  open: boolean;
  /** Close handler */
  onClose: () => void;
  /** Async detection handler. Return a Blob (image) or a data URL string. */
  onDetect: (file: File) => Promise<Blob | string>;
  /** Optional title */
  title?: string;
};

export default function DetectModal({
  open,
  onClose,
  onDetect,
  title = "Run detection",
}: DetectModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Track and revoke object URLs to avoid memory leaks
  const lastObjectUrl = useRef<string | null>(null);
  const setObjectUrl = (url: string) => {
    // Revoke previous URL (if any) before storing new one
    if (lastObjectUrl.current) URL.revokeObjectURL(lastObjectUrl.current);
    lastObjectUrl.current = url;
  };

  useEffect(() => {
    return () => {
      if (lastObjectUrl.current) URL.revokeObjectURL(lastObjectUrl.current);
    };
  }, []);

  function onSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResultUrl(null);
    setErrorMsg(null);

    if (f) {
      const url = URL.createObjectURL(f);
      setObjectUrl(url);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  }

  async function handleDetect() {
    if (!file) return;
    setLoading(true);
    setErrorMsg(null);
    setResultUrl(null);

    try {
      const out = await onDetect(file);
      if (typeof out === "string") {
        // data URL or remote URL
        setResultUrl(out);
      } else {
        // Blob returned -> turn into object URL
        const url = URL.createObjectURL(out);
        setObjectUrl(url);
        setResultUrl(url);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Detection failed. See logs.";
      setErrorMsg(message);
    } finally {
      setLoading(false);
    }
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-5xl rounded-2xl bg-neutral-900 text-white shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-neutral-800 px-6 py-4">
          <h2 className="text-xl font-semibold">{title}</h2>
          <button
            onClick={onClose}
            className="rounded-lg bg-neutral-800 px-3 py-1.5 hover:bg-neutral-700"
            aria-label="Close"
          >
            close
          </button>
        </div>

        {/* Body */}
        <div className="space-y-6 px-6 py-6">
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-neutral-800 px-4 py-2 hover:bg-neutral-700">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onSelect}
              />
              Choose image
            </label>

            <button
              onClick={handleDetect}
              disabled={!file || loading}
              className="rounded-xl bg-blue-600 px-4 py-2 disabled:cursor-not-allowed disabled:bg-blue-900 hover:bg-blue-500"
            >
              {loading ? "Detecting..." : "detect"}
            </button>

            {errorMsg && (
              <span className="text-sm text-red-400">{errorMsg}</span>
            )}
          </div>

          {/* Preview / Result */}
          <div className="grid gap-6">
            {/* Preview */}
            <section className="space-y-2">
              <h3 className="text-lg font-medium">
                {resultUrl ? "Result" : "Preview"}
              </h3>
              <div className="rounded-2xl bg-neutral-950 p-4">
                <div className="relative w-full">
                  {/* Keep the natural aspect by letting the image size itself.
                     Using next/image with fill requires a fixed container,
                     so here we pass width/height dynamically when possible. */}
                  {resultUrl ? (
                    <Image
                      src={resultUrl}
                      alt="Result"
                      className="h-auto w-full rounded-xl"
                      width={1600}
                      height={1200}
                      unoptimized
                      priority
                    />
                  ) : previewUrl ? (
                    <Image
                      src={previewUrl}
                      alt="Preview"
                      className="h-auto w-full rounded-xl"
                      width={1600}
                      height={1200}
                      unoptimized
                      priority
                    />
                  ) : (
                    <div className="flex h-48 items-center justify-center rounded-xl border border-neutral-800 text-neutral-400">
                      No image selected
                    </div>
                  )}
                </div>
              </div>
              {!resultUrl && (
                <p className="text-xs text-neutral-400">
                  Tip: PNG or JPG recommended. Your image stays local to this
                  request.
                </p>
              )}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
