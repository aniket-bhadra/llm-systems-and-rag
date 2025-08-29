import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import readlineSync from "readline-sync";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const history = [];

async function chatting(userProblem) {
  history.push({
    role: "user",
    parts: [{ text: userProblem }],
  });
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: history,
  });
  history.push({
    role: "model",
    parts: [{ text: response.text }],
  });
  console.log("\n");
  console.log(response.text);
}

async function main() {
  while (true) {
    const userProblem = readlineSync.question("Ask me anything---> ");
    await chatting(userProblem);
  }
}

main();
