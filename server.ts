import express from "express";
import * as googleTTS from "google-tts-api";
import path from "path";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API to generate free TTS via Google Translate API
  app.post("/api/tts", async (req, res) => {
    try {
      const { text, lang = "km" } = req.body;
      if (!text) {
        return res.status(400).json({ error: "Text is required" });
      }

      // Convert the text to audio base64 parts using google-tts-api
      // It handles texts longer than 200 characters natively by splitting
      const results = await googleTTS.getAllAudioBase64(text, {
        lang: lang,
        slow: false,
        host: "https://translate.google.com",
        timeout: 10000,
      });

      // return array of base64 chunks
      res.json({ results });
    } catch (error: any) {
      console.error("TTS generation error:", error);
      res.status(500).json({ error: error.message || String(error) });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
