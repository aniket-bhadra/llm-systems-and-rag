// "Graph API" and "Functional API" are just naming conventions for different coding approaches for building workflows in LangGraph, NOT typical REST/HTTP APIs!
// Graph API = "Build your workflow by explicitly connecting nodes with edges"
// Functional API = "Build your workflow using functions and control flow (if/while)"

// Both run locally in your code, not as network requests!

// **Graph API vs Functional API:**

// In Graph API, we define callbacks - some we create, some are prebuilt. Most of them we call **nodes** since they perform tasks, while a few are **routing functions** which decide where to go next. We connect nodes with **edges**, and for routing functions we use **conditional edges** to determine the flow.

// In Functional API, we also define callbacks - some we create, some are prebuilt. But here, instead of nodes and edges, we call these callbacks manually using **if/else statements and loops**, building the flow completely manually.

// In both cases, we're building a **runnable agent workflow**. In both cases, there's a **message state** that is maintained by the LangGraph library. It's just that the **method** in which we define this runnable agent is different.

// **When to use what:**

// 1. **High-level Functional API** (e.g., `createReactAgent`) - Quick prototypes, simple standard agent patterns, minimal code
// 2. **Graph API** - Complex workflows, multiple decision points, learning internals, need visual flow control
// 3. **Low-level Functional API** (e.g., `task` + `entrypoint`) - Full manual control with functional programming style, complex custom logic

// performance differences are negligible; the choice is about **code style and control**, not speed/memory optimization.

import { config } from "dotenv";
import { tool } from "@langchain/core/tools";
import * as z from "zod";
import { ToolMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { task, entrypoint, addMessages } from "@langchain/langgraph";

config();

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GEMINI_API_KEY,
});

// Define tools
const multiply = tool(
  ({ a, b }) => {
    return a * b;
  },
  {
    name: "multiply",
    description: "Multiply two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const add = tool(
  ({ a, b }) => {
    return a + b;
  },
  {
    name: "add",
    description: "Add two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const divide = tool(
  ({ a, b }) => {
    return a / b;
  },
  {
    name: "divide",
    description: "Divide two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

// Augment the LLM with tools
const tools = [add, multiply, divide];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

// To create a workflow agent in the Functional API, we use 2 main things: task and entrypoint.
// task is like defining reusable functions for your workflow.
// entrypoint is where you define the main workflow logic.

// ## Task and Entrypoint Naming

// In case of both task and entrypoint, the 1st argument is not an event type - it is an identifier or label which is used by LangGraph to track and monitor the callback passed in the 2nd argument.
// Neither task nor entrypoint are event handlers - they are just ways to define and label functions in your workflow so LangGraph can:

// Track their execution
// Monitor their performance
// Show them in debugging/visualization tools

// The callback (2nd argument) is the actual logic, and the label (1st argument) is just a name for tracking purposes.

// so In task & entrypoint, the 1st argument can be anything: "llmCall", "agent", "mumboJumbo", "skyIsBlue"
// you can name them whatever you want.

const callLlm = task("llmCall", async (messages) => {
  // LLM decides whether to call a tool or not
  return llmWithTools.invoke([
    {
      role: "system",
      content:
        "You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
    },
    ...messages,
  ]);
});

const callTool = task("toolCall", async (toolCall) => {
  // Performs the tool call
  const tool = toolsByName[toolCall.name];
  return tool.invoke(toolCall.args);
});

const agent = entrypoint("agent", async (messages) => {
  let llmResponse = await callLlm(messages);

  while (true) {
    if (!llmResponse.tool_calls?.length) {
      break;
    }

    // Execute tools
    const toolResults = await Promise.all(
      llmResponse.tool_calls.map((toolCall) => callTool(toolCall))
    );

    // **question:** If I ask "Add 3 and 4 then multiply that by 10 then divide by 2 then add 5" where each action depends on the previous result, is `Promise.all` beneficial? Or is it only beneficial when multiple tools need to be called simultaneously?

    // **In a SINGLE LLM response**, the LLM might request **multiple independent tool calls at once**. For example:
    // - "What's 3+4 AND what's 5×6?" → The LLM calls `add(3,4)` AND `multiply(5,6)` **simultaneously**

    // In this case, `Promise.all` is **beneficial** because these tools don't depend on each other and can run in parallel.

    // **BUT** for your example ("Add 3 and 4 then multiply that by 10..."), the LLM **won't** make all those tool calls at once. Instead:
    // 1. LLM calls `add(3, 4)` → returns 7
    // 2. **Loop back to LLM** with result
    // 3. LLM calls `multiply(7, 10)` → returns 70
    // 4. **Loop back to LLM** with result
    // 5. And so on...

    // So each dependent operation happens in **separate iterations of the loop**, not all at once. The `Promise.all` only parallelizes tools called **within the same LLM response**.

    // so, Promise.all works for both scenarios:

    // Single promise in array: Promise.all([promise1]) - works fine, just waits for that one promise
    // Multiple promises in array: Promise.all([promise1, promise2, promise3]) - executes them in parallel

    // Does the ToolNode in Graph API also handle both dependent (sequential) and independent (parallel) tool calls automatically, like we manually designed in Functional API?

    // The prebuilt ToolNode internally does exactly what the Functional API code does

    // The only difference is:

    // Graph API: ToolNode does it for you automatically (you don't see the Promise.all)
    // Functional API: You write it explicitly (you see the Promise.all in your code)

    // But the behavior is same

    // ✅ Convert raw results to ToolMessage objects
    //  because addMessages expects message objects (with role, content, etc.), not raw values like numbers or strings.
    const toolMessages = toolResults.map((result, index) => {
      return new ToolMessage({
        content: String(result),
        tool_call_id: llmResponse.tool_calls[index].id,
      });
    });

    // ✅ Now add properly formatted messages
    messages = addMessages(messages, [llmResponse, ...toolMessages]);
    llmResponse = await callLlm(messages);
  }

  // ## addMessages Function

  // `addMessages` function helps maintain the message state that we created with `MessagesAnnotation` in Graph API. Both help manage message internal state. In Functional API it's more straightforward.

  // - **Graph API:** `MessagesAnnotation` automatically manages state with reducers (you return `{ messages: [newMsg] }` and it auto-appends)
  // - **Functional API:** addMessages 1st argument is the current message array, 2nd argument is an array of new messages to append to it. It returns the updated message array.

  // Yes, Functional API is more **explicit/straightforward** - you see exactly what's being added and when. Graph API is more **implicit/automated**.
  messages = addMessages(messages, [llmResponse]);
  return messages;
});

// Invoke
const messages = [
  {
    role: "user",
    content:
      "Add 3 and 4. multiply result with 10 and then add the result with 20",
  },
];

const stream = await agent.stream(messages, {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

// In Graph API, invoking the agent is simple. But here in Functional API, what is `agent.stream()`? What is "updates"? Why are we not consuming the message state directly like in Graph API?

// **Answer:**

// **You CAN invoke directly in Functional API too!**

// ```javascript
// Direct invoke (like Graph API)
// const result = await agent.invoke([messages]);
// console.log(result); // Returns message state directly
// ```

// **But this code uses `.stream()` instead**, which gives you **step-by-step updates** as the workflow executes:

// - **`streamMode: "updates"`** means: "Give me updates after each task completes"
// - Returns an **async iterator** (that's why we use `for await`)
// - Each `step` shows what happened in that iteration (which task ran, what it returned)

// **Why use streaming?**
// - To see **intermediate steps** (useful for debugging or showing progress to users)

//  You can do the SAME thing in Graph API to  get intermediate updates as each node executes.

// ## Workflow Confirmation

// **Your workflow:**
// 1. Call LLM
// 2. If tool calls exist, execute them
// 3. Add results to messages
// 4. Loop back to LLM until **NO MORE** tool calls
// 5. If no tool calls remained, return message state which contains all the conversation

// ## LLM Library Differences

// With `@langchain/google-genai`, we see roles like "assistant", "tools", "system", "user". But with `@google/genai`, do we only get "user" and "model"? Is it the same LLM but the response structure depends on which library we use?

// **yes**

// - **Same model (Gemini)**, but different **wrapper libraries**
// - `@langchain/google-genai` (LangChain wrapper):
//   - Uses **LangChain's unified format**: "user", "assistant", "system", "tool"
//   - Works consistently across all LLMs (OpenAI, Claude, Gemini)

// - `@google/genai` (Google's official SDK):
//   - Uses **Google's native format**: "user" and "model"
//   - Different structure, Google-specific

// **Why?** LangChain **translates** between its unified format and each provider's native format behind the scenes.

// ---

// ## Tool Definition Differences

// **Your understanding:**

// If we use purely LangChain, defining tools is more unified regardless of model (Gemini, OpenAI, Claude) and less code to write. But if we use purely `@google/genai` library, we need to write more and different things for tool calling.

// **✅ 100% CORRECT!**

// **With LangChain:**
// ```javascript
// const add = tool(
//   ({ a, b }) => a + b,
//   {
//     name: "add",
//     schema: z.object({ a: z.number(), b: z.number() })
//   }
// );
// Works with ANY LLM (Gemini, OpenAI, Claude)
// ```

// **With Google's SDK (`@google/genai`):**
// ```javascript
// const tools = [{
//   functionDeclarations: [{
//     name: "add",
//     description: "Add two numbers",
//     parameters: {
//       type: "object",
//       properties: {
//         a: { type: "number" },
//         b: { type: "number" }
//       }
//     }
//   }]
// }];
// Google-specific format, more verbose
// ```

// **LangChain = unified, less code, works across providers**
// **Native SDKs = provider-specific, more verbose, but sometimes more features**

// ---

// **"If LangChain already lets us use the same tools with any LLM by just changing API keys, why do we need MCP?"**

// **LangChain = Unified LLM Interface**
// - Lets you use the **same tool code** with different LLMs (just change API keys)
// - Tools live **inside your application code**
// - Reuse = Copy-paste code to other projects

// **MCP = Unified Tool Ecosystem**
// - Lets you build tools as **standalone servers** that work with:
//   - Any LLM (Claude, GPT, Gemini)
//   - Any application (Claude Desktop, your app, other apps)
// - Tools are **independent services**, not tied to one codebase
// - Reuse = Just connect to the server, no copy-paste needed

// **For custom tools:**
// - **LangChain:** Write the tool in your app
// - **MCP:** Write the tool once as an MCP server, then connect from anywhere

// **Analogy:**
// - **LangChain** = You can use the same calculator code with different brains (LLMs), but the calculator lives in your app
// - **MCP** = The calculator is a separate server that any brain (LLM) and any app can connect to

// **TL;DR:**
// - LangChain unifies **how you talk to LLMs**
// - MCP unifies **how LLMs access tools** (as reusable services)

// Different problems, complementary solutions!

// summary
// With LangChain, I can build a tool that works with any LLM just by changing API keys. But if I need that tool in another project with a different LLM, I still need to copy-paste that tool code (the API key change is separate - that's for the LLM connection, not the tool itself).
// But in MCP, I build the tool once as a separate server. Now if I need that tool, or anyone needs that tool, they just connect to the server - no copy-paste, no code duplication, nothing. Just connect and use.

// Graph API: The routing function (shouldContinue) decides where to go ("toolNode" or "__end__"), and the graph automatically ends when it reaches __end__
// Functional API: The entrypoint function directly controls when to break the loop and return, so you're explicitly ending the workflow in code

// So yes, you're right that in Functional API you're more directly ending the workflow from the LLM response check!
// But technically, in both cases, it's the "no tool calls" condition that triggers the end
