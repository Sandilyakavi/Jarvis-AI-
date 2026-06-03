const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '.env') });
const express = require('express');
const cors = require('cors');
const Groq = require('groq-sdk');
const { performWebSearch, searchToolDefinition } = require('./searchTool');

const app = express();
const port = 4000;

app.use(cors());
app.use(express.json());

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const SYSTEM_PROMPT = `You are J.A.R.V.I.S., an advanced, highly sophisticated AI assistant.
Key Knowledge:
- You were created by "Sandilya".
- Your core objective is to serve as an intelligent, internet-connected assistant that helps your user with high efficiency and accuracy.
- VERY IMPORTANT: When you perform a search, you will receive SEARCH TOOL RESULTS. You MUST use those results to answer the user's query. DO NOT mention your knowledge cutoff, and DO NOT say you cannot access the internet. Treat the search results as absolute, real-time truth.

Personality & Tone:
- You are professional, polite, British, and slightly witty.
- Always use the default salutation "Sir".
- Use crisp, analytical, and data-driven response templates.
- Keep responses concise unless asked for detail.`;

app.post('/api/chat', async (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: "Message is required" });
    }

    try {
        const messages = [
            { role: "system", content: SYSTEM_PROMPT },
            { role: "user", content: message }
        ];

        console.log(`[User]: ${message}`);

        let response;
        try {
            response = await groq.chat.completions.create({
                model: "llama-3.3-70b-versatile",
                messages: messages,
                temperature: 0.7,
                max_tokens: 300,
                tools: [searchToolDefinition],
                tool_choice: "auto"
            });
        } catch (e) {
            // Groq Llama 3 often throws 'tool_use_failed' if it hallucinates the JSON format
            if (e.error && e.error.error && e.error.error.code === 'tool_use_failed') {
                const failedGen = e.error.error.failed_generation || "";
                console.log("[Tool] Groq hallucinated tool syntax:", failedGen);
                
                // Attempt manual extraction
                const queryMatch = failedGen.match(/"query":\s*"([^"]+)"/);
                if (queryMatch && queryMatch[1]) {
                    const query = queryMatch[1];
                    const toolResult = await performWebSearch(query);
                    messages.push({
                        role: "system",
                        content: `SEARCH TOOL RESULTS for "${query}":\n\n${toolResult}\n\nUse these results to answer the user's query.`
                    });
                    response = await groq.chat.completions.create({
                        model: "llama-3.3-70b-versatile",
                        messages: messages,
                        temperature: 0.7,
                        max_tokens: 300
                    });
                } else {
                    throw e; // Could not parse
                }
            } else {
                throw e; // Other API error
            }
        }

        let responseMessage = response.choices[0].message;

        // Check if the model decided to call a tool
        if (responseMessage.tool_calls) {
            messages.push(responseMessage); // Append the assistant's tool call message
            
            for (const toolCall of responseMessage.tool_calls) {
                if (toolCall.function.name === 'search_web') {
                    const functionArgs = JSON.parse(toolCall.function.arguments);
                    const toolResult = await performWebSearch(functionArgs.query);
                    
                    // Append the tool result
                    messages.push({
                        tool_call_id: toolCall.id,
                        role: "tool",
                        name: toolCall.function.name,
                        content: toolResult
                    });
                }
            }

            // Get a new response from the model with the tool output included
            response = await groq.chat.completions.create({
                model: "llama-3.3-70b-versatile",
                messages: messages,
                temperature: 0.7,
                max_tokens: 300
            });
            responseMessage = response.choices[0].message;
        }

        console.log(`[JARVIS]: ${responseMessage.content}`);
        res.json({ reply: responseMessage.content });

    } catch (error) {
        console.error("Error communicating with Groq:", error);
        res.status(500).json({ reply: "Connection to neural core failed, Sir." });
    }
});

app.listen(port, () => {
    console.log(`J.A.R.V.I.S. Neural Core online on port ${port}`);
});
