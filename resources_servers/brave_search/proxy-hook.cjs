// Node's global fetch (undici) does not honor HTTP_PROXY/HTTPS_PROXY on its own
// (unlike Python's requests/httpx). Loaded via NODE_OPTIONS="--require .../proxy-hook.cjs"
// to route it through the configured proxy when spawning the Brave MCP server.
const { ProxyAgent, setGlobalDispatcher } = require("undici");

const proxyUrl = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
if (proxyUrl) {
  setGlobalDispatcher(new ProxyAgent(proxyUrl));
  console.error(`[proxy-hook] routing fetch through ${proxyUrl.replace(/:[^:@]*@/, ":***@")}`);
}
