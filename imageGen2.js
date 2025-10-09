import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import * as fs from "node:fs";

async function main() {
  const ai = new GoogleGenAI({ apikey: process.env.GEMINI_API_KEY });

  const response = await ai.models.generateImages({
    model: "imagen-4.0-generate-001",
    prompt: "Create a picture of a logo written hungrydip",
    config: {
      numberOfImages: 2,
    },
  });

  let idx = 1;
  for (const generatedImage of response.generatedImages) {
    let imgBytes = generatedImage.image.imageBytes;
    const buffer = Buffer.from(imgBytes, "base64");
    fs.writeFileSync(`imagen-${idx}.png`, buffer);
    idx++;
  }
}

main();

// When you set numberOfImages: 2, the API returns 2 different image candidates (variations). The response.generatedImages is an array of 2 objects, each containing one generated image.