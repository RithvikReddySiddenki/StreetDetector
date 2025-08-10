StreetDetector is a Next.js web application for real-time YOLOv5 object detection — running entirely in your browser using ONNX Runtime Web.
No backend server is required; the detection happens client-side using WebAssembly.

Features
Client-side inference with YOLOv5m in ONNX format — no Python backend needed.

Upload any image and detect objects directly in the browser.

Uses ONNX Runtime Web with WebAssembly for fast, portable execution.

Letterbox preprocessing to preserve aspect ratio when resizing images.

Non-Maximum Suppression (NMS) to filter overlapping detections.

Draws bounding boxes and labels directly on the original image without dimming.

Responsive, large preview canvas for better visual clarity.

Displays detection scores with class names.

Built with Next.js 13+ App Router.

Deployed on Vercel for instant public access.

Footer credit for Rithvik Reddy Siddenki.

Tech Stack
Frontend: Next.js, TypeScript, Tailwind CSS

Model Execution: ONNX Runtime Web (WASM backend)

Model: YOLOv5m (exported to ONNX)

Deployment: Vercel
