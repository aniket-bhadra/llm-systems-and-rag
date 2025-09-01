import "dotenv/config";
import { GoogleGenAI, Modality } from "@google/genai";
import * as fs from "node:fs";

async function main() {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  const prompt = "Create a picture of a logo written hungrydip";

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-image-preview",
    contents: prompt,
  });
  for (const part of response.candidates[0].content.parts) {
    if (part.text) {
      console.log(part.text);
    } else if (part.inlineData) {
      const imageData = part.inlineData.data;
      const buffer = Buffer.from(imageData, "base64");
      fs.writeFileSync("logo.png", buffer);
      console.log("Image saved as logo.png");
    }
  }
}

main();
