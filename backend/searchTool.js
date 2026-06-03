const cheerio = require("cheerio");

async function performWebSearch(query) {
    try {
        console.log(`[Tool] Performing web search for: "${query}"`);
        const res = await fetch(`https://search.yahoo.com/search?p=${encodeURIComponent(query)}`, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        });
        const html = await res.text();
        const $ = cheerio.load(html);
        const results = [];
        
        $(".algo").each((i, el) => {
            if (i < 5) {
                const title = $(el).find(".title").text().trim();
                const desc = $(el).find(".compText").text().trim();
                if (title && desc) {
                    results.push(`Title: ${title}\nSnippet: ${desc}`);
                }
            }
        });

        if (results.length === 0) {
            return "No relevant search results found on the web.";
        }

        const topResults = results.join('\n\n');
        return topResults;
    } catch (error) {
        console.error("[Tool] Web search failed:", error);
        return "An error occurred while attempting to search the web. Please inform the user that the internet connection might be unstable.";
    }
}

const searchToolDefinition = {
    type: "function",
    function: {
        name: "search_web",
        description: "Search the LIVE internet for real-time information, facts, news, sports scores, and current events. Always use this tool for any question regarding current events.",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "The exact search query to look up on the web."
                }
            },
            required: ["query"]
        }
    }
};

module.exports = {
    performWebSearch,
    searchToolDefinition
};
