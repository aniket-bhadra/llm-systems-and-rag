import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import readlineSync from "readline-sync";

function sum({ num1, num2 }) {
  return num1 + num2;
}

function isPrime({ num }) {
  for (let i = 2; i <= Math.sqrt(num); i++) {
    if (num % i === 0) {
      return false;
    }
  }
  return true;
}

async function ageDetector({ name }) {
  try {
    const res = await fetch(`https://api.agify.io/?name=${name}`);
    const data = await res.json();
    //  console.log(data.age);
    return data.age;
  } catch (error) {
    console.log(`error detecting age-- ${error}`);
  }
}

// console.log(ageDetector("robin"));

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const history = [];

const sumDeclaration = {
  name: "sum",
  description: "get the sum of 2 numbers",
  parameters: {
    type: "OBJECT",
    properties: {
      num1: {
        type: "NUMBER",
        description: "it will be first number for sum ex:10",
      },
      num2: {
        type: "NUMBER",
        description: "it will be second number for sum ex:7",
      },
    },
    required: ["num1", "num2"],
  },
};

const isPrimeDeclaration = {
  name: "isPrime",
  description: "check if the number prime or not",
  parameters: {
    type: "OBJECT",
    properties: {
      num: {
        type: "NUMBER",
        description: "it will be the number to find prime or not ex:7",
      },
    },
    required: ["num"],
  },
};

const ageDetectorDeclaration = {
  name: "ageDetector",
  description: "return age based on the name we provide",
  parameters: {
    type: "OBJECT",
    properties: {
      name: {
        type: "STRING",
        description: "it will be the name to find the age ex: rohit",
      },
    },
    required: ["name"],
  },
};

const availableTools = {
  sum,
  isPrime,
  ageDetector,
};

async function runAgent(userProblem) {
  history.push({
    role: "user",
    parts: [{ text: userProblem }],
  });

  while (true) {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: history,
      config: {
        tools: [
          {
            functionDeclarations: [
              sumDeclaration,
              isPrimeDeclaration,
              ageDetectorDeclaration,
            ],
          },
        ],
        systemInstruction: `you are an AI Agent,you have access 3 available tools - which are to find sum of two numbers, checking is a number prime or not, take an name and return age.
        
        so whenever user queries related to this, do take help of these tools but if user query is for a general Question then you can answer it directly
        `,
      },
    });

    if (response.functionCalls && response.functionCalls.length > 0) {
      const { name, args } = response.functionCalls[0];
      // console.log(response.functionCalls);
      console.log(response.functionCalls[0]);

      const fn = availableTools[name];
      const preResult = fn(args);
      const result = preResult instanceof Promise ? await preResult : preResult;

      const functionResponsePart = {
        name: name,
        response: {
          result: result,
        },
      };

      history.push({
        role: "model",
        parts: [
          {
            functionCall: response.functionCalls[0],
          },
        ],
      });
      history.push({
        role: "user",
        parts: [
          {
            functionResponse: functionResponsePart,
          },
        ],
      });
    } else {
      history.push({
        role: "model",
        parts: [{ text: response.text }],
      });
      console.log(response.text);
      break;
    }
  }
}

async function main() {
  while (true) {
    const userProblem = readlineSync.question("Ask me anything---> ");
    await runAgent(userProblem);
  }
}
main();


// --------------------------------------------notes--------------------

// ## Where You Define the Structure:

// ```javascript
// const sumDeclaration = {
//   name: "sum",                                     // 👈 YOU define the function name here
//   description: "get the sum of 2 numbers",
//   parameters: {
//     type: "OBJECT",                                // 👈 Structure will be an OBJECT
//     properties: {
//       num1: {                                      // 👈 YOU define "num1" key here
//         type: "NUMBER",
//         description: "it will be first number for sum ex:10",
//       },
//       num2: {                                      // 👈 YOU define "num2" key here
//         type: "NUMBER",
//         description: "it will be second number for sum ex:7",
//       },
//     },
//     required: ["num1", "num2"],
//   },
// };
// ```

// ## This Creates the Structure:

// ```javascript
// {
//   name: "sum",     // ← From sumDeclaration.name
//   args: {          // ← From parameters.type: "OBJECT"
//     num1: 5,       // ← From properties.num1
//     num2: 3        // ← From properties.num2
//   }
// }
// ```

// **So:**
// - **Structure definition** = `sumDeclaration` (YOU write this)
// - **Structure population** = AI fills in the values based on user query

// The `sumDeclaration` is the **blueprint/template** that tells AI: "When you call this function, use this exact structure!" 🎯

// -----------------------------------------------------

// !! The Problem in the current code:
// In the else block, the model might return:

// Text parts
// Thought signatures (reasoning process)
// Other metadata or structured data

// But we are only capturing text, so the warning says: "Hey, there are other non-text parts in the response that you're ignoring!"

// else {
//   const fullResponse = response.candidates?.[0]?.content || {
//     parts: [{ text: response.text }]
//   };
  
//   history.push({
//     role: "model",
//     parts: fullResponse.parts,  // ALL parts (text + non-text)
//   });
  
//   console.log(response.text);
//   break;
// }