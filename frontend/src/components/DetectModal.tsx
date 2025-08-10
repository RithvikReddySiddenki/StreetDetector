// src/components/DetectModal.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Props = {
  open: boolean;
  onClose: () => void;
};

const INFERENCE_URL = process.env.NEXT_PUBLIC_INFERENCE_URL || process.env.INFERENCE_URL;

export default function DetectModal({ open, onClose }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) {
      // reset state when modal closes
      setFile(null);
      setLoading(false);
      setError(null);
      setResultUrl(null);
      if (inputRef.current) inputRef.current.value = "";
    }
  }, [open]);

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setResultUrl(null);
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const onDetect = async () => {
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    if (!INFERENCE_URL) {
      setError("INFERENCE_URL is not configured.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResultUrl(null);

      const form = new FormData();
      form.append("file", file);

      // Ask backend for rendered image
      const url = `${INFERENCE_URL}?render=true`;
      const res = await fetch(url, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Request failed: ${res.status}`);
      }

      // Backend returns image bytes when render=true
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      setResultUrl(objectUrl);
    } catch (err: any) {
      setError(err?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Card */}
          <motion.div
            className="w-full max-w-2xl rounded-2xl bg-[#0b0f17] border border-white/10 shadow-xl overflow-hidden"
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 30, opacity: 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 24 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
              <div className="font-semibold text-white">Run detection</div>
              <button
                onClick={onClose}
                className="rounded-md px-3 py-1 text-sm bg-white/10 hover:bg-white/15"
              >
                close
              </button>
            </div>

            {/* Body */}
            <div className="p-6 space-y-6">
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <div>
                  <label className="block text-sm text-white/80 mb-2">
                    Choose image
                  </label>
                  <div className="rounded-xl border border-dashed border-white/20 bg-white/5 p-4">
                    <input
                      ref={inputRef}
                      type="file"
                      accept="image/*"
                      onChange={onPick}
                      className="block w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-white file:px-4 file:py-2 file:text-black file:font-semibold hover:file:bg-white/90"
                    />
                    <p className="mt-3 text-xs text-white/60">
                      PNG or JPG recommended. Your image stays local to this request.
                    </p>
                  </div>

                  <button
                    onClick={onDetect}
                    disabled={loading}
                    className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 font-semibold hover:bg-white/90 disabled:opacity-60"
                  >
                    {loading ? "detecting..." : "detect"}
                  </button>

                  {error && (
                    <p className="mt-3 text-sm text-red-400">{error}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm text-white/80 mb-2">
                    Result
                  </label>

                  <div className="relative rounded-xl border border-white/10 bg-black/20 aspect-[4/3] overflow-hidden flex items-center justify-center">
                    {!resultUrl && !loading && (
                      <p className="text-white/50 text-sm">No result yet</p>
                    )}

                    {loading && (
                      <motion.div
                        className="h-14 w-14 rounded-full border-4 border-white/20 border-t-white"
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                      />
                    )}

                    {resultUrl && !loading && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={resultUrl}
                        alt="detection result"
                        className="h-full w-full object-contain"
                      />
                    )}
                  </div>
                </div>
              </div>

              <p className="text-xs text-white/50">
                Tip: if you want JSON instead of an image, call the backend without
                <code className="mx-1 rounded bg-white/10 px-1.5 py-0.5">render=true</code>.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
