StreetDetector is a high-performance Next.js web application for real-time YOLOv5 object detection, executed entirely in the browser using ONNX Runtime Web.
By leveraging WebAssembly, the application performs all inference client-side, eliminating the need for a backend server and ensuring low-latency, private, and portable execution.

Key Features
Fully Client-Side Inference – Runs YOLOv5m in ONNX format directly in the browser, with no Python backend required.

Instant Object Detection – Upload an image and detect objects immediately, without sending data to a server.

Optimized Execution – Uses ONNX Runtime Web with WebAssembly for fast, cross-platform performance.

Letterbox Preprocessing – Maintains the original aspect ratio when resizing images for detection.

Non-Maximum Suppression (NMS) – Reduces duplicate or overlapping detections for cleaner results.

Accurate Visualization – Draws bounding boxes and class labels directly over the original image without dimming or distortion.

Responsive UI – Large, adaptive preview canvas for enhanced visual clarity.

Detailed Results – Displays both detection scores and class names for identified objects.

Modern Framework – Built with Next.js 13+ App Router for scalability and maintainability.

Seamless Deployment – Hosted on Vercel for instant global access.

Technology Stack
Frontend Framework: Next.js, TypeScript, Tailwind CSS

Model Execution: ONNX Runtime Web (WASM backend)

Model: YOLOv5m (exported to ONNX)

Deployment Platform: Vercel
