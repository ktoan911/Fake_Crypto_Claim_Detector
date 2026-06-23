const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

// ── Types matching api_server.py `/claims/stats` output ────────────────────

export interface RecentClaim {
    claim: string;
    verdict: string;
    checked_at: string;
}

export interface Stats24h {
    "đúng": number;
    "sai": number;
    "chưa chắc chắn": number;
    "percent_đúng": number;
    "percent_sai": number;
}

export interface ClusterItem {
    topic_label: string;
    claims: string[];
    size: number;
    representative_claim?: string;
    cluster_content?: string;
    cluster_id?: number;
}

export interface RawClusterItem {
    topic_label?: string;
    claims?: unknown;
    size?: number;
    representative_claim?: string;
    cluster_content?: string;
    cluster_id?: number;
}

export interface ClusterPayload {
    clusters?: RawClusterItem[];
    num_input_claims?: number;
    num_clusters?: number;
    model_name?: string;
    error?: string;
}

export interface ClaimsStats {
    date: string;
    timestamp: string;
    recent_claims: RecentClaim[];
    stats_24h: Stats24h;
    /** key = "YYYY-MM-DD", value = count */
    daily_total: Record<string, number>;
    daily_false: Record<string, number>;
    cluster: RawClusterItem[] | ClusterPayload | Record<string, unknown>;
}

// ── Fetch functions ─────────────────────────────────────────────────────────

export async function fetchClaimsStats(date?: string): Promise<ClaimsStats> {
    const url = date
        ? `${API_URL}/claims/stats?date=${date}`
        : `${API_URL}/claims/stats`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`fetchClaimsStats failed: ${res.status}`);
    return res.json();
}

// ── Types matching api_server.py `/crawler/info` output ────────────────────

export interface CrawlByDay {
    day: string;
    total_crawl: number;
}

export interface CrawlSourceItem {
    name: string;
    value: number;
}

export interface CrawlerInfo {
    crawl_by_day: CrawlByDay[];
    per_source: CrawlSourceItem[];
    total_saved?: number;
}

export async function fetchCrawlerInfo(days?: number): Promise<CrawlerInfo> {
    const url = days
        ? `${API_URL}/crawler/info?days=${days}`
        : `${API_URL}/crawler/info`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`fetchCrawlerInfo failed: ${res.status}`);
    return res.json();
}

export async function verifyClaim(claim: string) {
    const res = await fetch(`${API_URL}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim }),
    });
    if (!res.ok) throw new Error(`verifyClaim failed: ${res.status}`);
    return res.json();
}
