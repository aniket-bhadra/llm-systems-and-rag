# How LLMs Actually Work - A Clear Explanation

## LLMs Don't Store Exact Answers

LLMs don't work like traditional databases where they store exact answers and fetch them when queried. They don't store "the weather is 25°C in Mumbai" and retrieve it when you ask about weather. Instead, they work by recognizing patterns.

## Pattern Recognition, Not Database Matching

Think of it like this: if we ask humans what comes next in this sequence:
```
500
1000
1500
2000
??
```

Even if you've never seen this exact problem before, you can answer "2500" because you recognize the pattern - each number increases by 500. You understand the pattern and generate the next number accordingly.

Similarly, LLMs recognize patterns and generate answers. They don't match your query to a database and fetch results. Instead, they try to understand the pattern in your input and predict what should come next.

## How LLMs Generate Responses

When you say "Hi, how are you?" to an LLM, here's what actually happens:

1. It takes the input "Hi, how are you"
2. It tries to predict the next word based on patterns it learned during training
3. "Hi, how are you" → "Hi, how are you, I"
4. "Hi, how are you, I" → "Hi, how are you, I am" 
5. "Hi, how are you, I am" → "Hi, how are you, I am fine"
6. It extracts the generated part "I am fine" and sends it to you

## Tokenization: Converting Words to Numbers

Every LLM has a tokenization process where each word gets assigned a number:
- "hi" → 12194
- "how" → 11  
- "are" → 7
- "you" → 65

Each LLM has its own different number list. For the sentence "hi how are you", this creates 4 tokens. These numbers then go to the LLM model.

In tokenization, we can assign 1 character to 1 token, or 5 characters to 1 token - it varies.

## The Prediction Process

The LLM receives these token numbers and tries to find patterns to predict the next number:

```
[12194, 11, 7, 65] → LLM predicts → [12194, 11, 7, 65, 88]
[12194, 11, 7, 65, 88] → LLM predicts → [12194, 11, 7, 65, 88, 99]  
[12194, 11, 7, 65, 88, 99] → LLM predicts → [12194, 11, 7, 65, 88, 99, 256]
```

This process continues until completion:
```
[12194, 11, 7, 65, 88, 99, 256, 8524, 97425]
```

Since the LLM has a list mapping each number back to words, this becomes:
"Hi how are you, I am fine"

The model extracts "I am fine" and sends it to you.

LLMs count **both input tokens + generated output tokens** for total token calculation, not just the generated tokens.

For example: If you input "Hi how are you" (4 tokens) and the LLM generates "I am fine thank you" (5 tokens), the total token count would be 9 tokens (4 input + 5 output).

## Why It's Called "Generative AI"

This is called generative AI because it generates answers by recognizing patterns, not by matching databases and fetching stored results. It matches patterns and generates answers, even if it has never seen that exact question before.

**GPT stands for: Generative Pre-trained Transformer**
- **Generative**: It generates responses
- **Pre-trained**: It's already trained on data so it can identify patterns based on what it learned during training
- **Transformer**: It transforms input (like text prompts) into output (like images or text responses)

## Why Different Answers for the Same Question?

Consider this sequence: 1, 2, 4, ??

The next number could be:
- **8** (doubling each number: 1×2=2, 2×2=4, 4×2=8)
- **7** (adding incrementally: 1+1=2, 2+2=4, 4+3=7)

The same question can have different answers depending on which pattern the LLM identifies. For the same token numbers, it might find different patterns, leading to different answers for the same question.

How the LLM finds patterns and predicts the next number depends on many factors: previous chat history, user context, probability calculations, pre-trained data, and many other elements that help identify patterns and generate the next word.

## Limitations: What LLMs Can and Cannot Do

When you give "2+2" to an LLM, it doesn't actually perform the calculation. It predicts the answer based on patterns it has seen during training. Since it has been trained so well on basic math, it can predict "4" accurately without calculating.

However, if you give it:
```
6314758517471685 × 215415544414 = ?
```

It cannot give the correct answer because:
1. The model hasn't seen this exact data during training
2. It doesn't have arithmetic computational power to actually calculate

For questions it wasn't trained on, it can only identify patterns and predict based on training data. If it hasn't seen a pattern, it cannot generate the next word on its own.

## External Tools for Complex Tasks

For complex calculations, LLMs generate code to solve the problem. But since they can't run code themselves, they send this code to external tools. The external tool runs the code, gets the answer, and sends it back to the model, which then gives you the result.

Similarly, when you ask "What's the temperature in Mumbai today?", the LLM doesn't have this real-time data from its training. So it uses external tools that actually know Mumbai's current temperature, gets that information, and returns it to you.

## The Core Truth

A pure LLM model can only give you answers by prediction, and it predicts based on the data it was trained on. Everything else requires external tools and integrations.

### Context

Pure LLM models do not store data, so whatever we give as input for them is context. So if I tell "my name is Virat", then in the next question I ask "what is my name", it does not store that information - it will reply "I don't know". But if I would have "I'm Virat, tell me my name" so now it can say that because for it this is its context - the input is its context.

But when we chat with LLMs inside chatboxes, in that case whatever we ask, behind the scenes the chatbox takes all the previous questions and responses in that chatbox and inserts it to the model with the current question I ask, so that model actually gets the context of whatever we are conversing. So now the LLM model can identify the whole pattern and generate the next words based on that context perfectly.

Like this way it can perfectly tell what is my name.

```javascript
async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [
      {
        role: "user",
        parts: [{ text: "hi, I'm Virat" }],
      },
      {
        role: "model",
        parts: [{ text: "Hi Virat! It's nice to meet you." }],
      },
      {
        role: "user",
        parts: [{ text: "can you tell me my name" }],
      },
    ],
  });
  console.log(response.text);
}
```

But if we omit the context part:

```javascript
async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [
      {
        role: "user",
        parts: [{ text: "can you tell me my name" }],
      },
    ],
  });
  console.log(response.text);
}
```

Then it tells "sorry I cannot tell you personal information about you".
