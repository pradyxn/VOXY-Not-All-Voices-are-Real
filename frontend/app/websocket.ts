export function getWebSocketUrl(apiUrl: string, pageProtocol: string): string {
  if (!apiUrl) throw new Error("Backend URL is missing. Set NEXT_PUBLIC_API_URL and rebuild the frontend.");
  let backendUrl: URL;
  try {
    backendUrl = new URL(apiUrl);
  } catch {
    throw new Error("Backend URL is invalid. Use http://localhost:8000 or an https:// backend URL.");
  }
  if (backendUrl.protocol !== "http:" && backendUrl.protocol !== "https:") {
    throw new Error("Backend URL must use http:// or https://.");
  }
  if (pageProtocol === "https:" && backendUrl.protocol !== "https:") {
    throw new Error("Browser blocked insecure ws:// from an HTTPS page. Configure NEXT_PUBLIC_API_URL with https://.");
  }
  backendUrl.protocol = backendUrl.protocol === "https:" ? "wss:" : "ws:";
  backendUrl.pathname = "/ws/analyze";
  backendUrl.search = "";
  backendUrl.hash = "";
  return backendUrl.toString();
}
